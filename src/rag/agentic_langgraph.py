import asyncio
import json
import torch
import gc
import traceback
import uuid
from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, StateSnapshot
from langgraph.graph import add_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from qdrant_client import models
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

# ============================================================
# === Core Graph State Definition
# ============================================================
def add_list(left, right):
    if right == "CLEAR":
        return []
    if left is None:
        left = []
    if right is None:
        right = []
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right

class AgenticRAGState(TypedDict):
    query: str
    relevant_node_ids: Annotated[List[str], add_list]
    relevant_functions: Annotated[List[str], add_list]
    relevant_docs: Annotated[List[Document], add_list]
    messages: Annotated[list, add_messages]
    tool_log: Annotated[List[dict], add_list]


# ============================================================
# === Filter Builder Tool
# ============================================================
class FilterBuilderInput(BaseModel):
    query: str = Field(..., description="Natural language description of what to filter for")


async def build_qdrant_metadata_type_filter(query: str) -> models.Filter:
    """
    Build a Qdrant-compatible Filter object from a natural-language description.
    This enables agents to generate appropriate search filters dynamically.
    """

    q = query.lower().strip()

    matched_types = []

    if "import" in q:
        matched_types.append("import_name")
    if "docstring" in q:
        matched_types.append("function_docstring")
    if "class" in q:
        matched_types.append("class_def")
    if "function" in q:
        matched_types.extend(["function_code", "function_name"])
    if "issue" in q or "bug" in q:
        matched_types.extend(["issue_title", "issue_body"])
    if "pull request" in q or "pr" in q:
        matched_types.extend(["pr_title", "pr_body"])
    if "cluster" in q:
        matched_types.append("semantic_cluster")

    if not matched_types:
        print(f"[DEBUG] No matching keywords found in query: '{query}' → returning empty filter.")
        return models.Filter(must=[]).model_dump()

    condition = models.FieldCondition(
        key="metadata.type",
        match=models.MatchAny(any=matched_types),
    )

    qdrant_filter = models.Filter(must=[condition])

    #print(f"[DEBUG] Built Qdrant filter for '{query}': {qdrant_filter}")

    return qdrant_filter.model_dump()


# ============================================================
# === Qdrant Wrapper
# ============================================================
class QdrantSearchWrapper:
    """Wraps the MCP qdrant_search tool to extract documents and node IDs into state."""

    def __init__(self, tool):
        self.tool = tool
        self.name = tool.name
        self.description = tool.description
        self.args_schema = getattr(tool, "args_schema", None)

    async def ainvoke(self, args, **kwargs):
        """
        Run the wrapped MCP tool asynchronously and enrich the result.
        """
        #print(f"[DEBUG] Calling wrapped qdrant_search with args={args}")
        try:
            result = await self.tool.ainvoke(args, **kwargs)
        except Exception as e:
            # Handle validation / network errors
            print(f"[ERROR] qdrant_search failed: {e}")
            print("[FALLBACK] Retrying with empty metadata_filter...")

            # Attempt fallback with a safe empty filter
            fallback_args = dict(args)
            fallback_args["metadata_filter"] = {}
            try:
                result = await self.tool.ainvoke(fallback_args, **kwargs)
                print("[FALLBACK] Retry succeeded.")
            except Exception as e2:
                print(f"[ERROR] Fallback qdrant_search also failed: {e2}")
                # Return a safe empty response
                result = []

        if isinstance(result, str):
            try:
                result = json.loads(result)
                #print(f"[DEBUG] Parsed JSON string into {len(result)} results.")
            except json.JSONDecodeError:
                print("[WARN] Could not parse qdrant_search result as JSON.")
                result = []

        if not isinstance(result, list):
            print(f"[WARN] Unexpected qdrant_search return type: {type(result)}")
            result = []
        # Convert raw dicts → LangChain Documents
        docs = [Document(page_content=r["content"], metadata=r["metadata"]) for r in result]

        # Extract node IDs 
        node_ids = set()
        for doc in docs:
            meta = doc.metadata
            doc_type = meta.get("type", "").lower()
            raw_id = meta.get("node_id")

            if not doc_type or not raw_id:
                continue

            # 1. Always extract the global_id from metadata
            try:
                node_prefix = doc_type.split('_')[0].upper()
                global_id = f"{node_prefix}:{raw_id}"
                node_ids.add(global_id)
            except Exception:
                print(f"[WARN] Could not parse global_id from doc_type: {doc_type}")
                continue

        enriched_state = {
            "relevant_docs": docs,
            "relevant_node_ids": sorted(list(node_ids)),
        }

        #print(f"[DEBUG] Wrapped qdrant_search returned {len(docs)} docs and {len(node_ids)} node IDs.")
        return result, enriched_state

# ===========================================================
class GetNodeInfoInput(BaseModel):
    node_id: str = Field(..., description="The global node ID, e.g. FUNC:12345")
    field: str = Field(..., description="The property name to retrieve (must be a string).")


# ============================================================
# === Basic Tool Node
# ============================================================
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    async def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        updated_state = {"tool_log": inputs.get("tool_log", []).copy()}

        for tool_call in message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name not in self.tools_by_name:
                warn_msg = f"[WARN] LLM requested unknown tool '{tool_name}'. Skipping this call."
                print(warn_msg)

                updated_state["tool_log"].append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_summary": "Invalid tool name"
                })

                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )
                continue
            
            tool = self.tools_by_name[tool_name]

            #print(f"[DEBUG] Executing tool: {tool_name} with args={tool_args}")
            try:
                tool_result = await tool.ainvoke(tool_args)
            except Exception as e:
                err_msg = f"[ERROR] Tool '{tool_name}' failed with exception: {e}"
                print(err_msg)
                traceback.print_exc()

                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": str(e)}),
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )

                updated_state["tool_log"].append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_summary": f"Error: {e}"
                })

                continue
            
            # Handle wrapped tools that return (result, state_updates)
            state_updates = {}
            if isinstance(tool_result, tuple) and len(tool_result) == 2:
                result, state_updates = tool_result
                updated_state.update(state_updates)
            else:
                result = tool_result

            updated_state["tool_log"].append({
                "tool": tool_name,
                "args": tool_args,
                "result_summary": (
                    f"{len(state_updates.get('relevant_docs', []))} docs"
                    if "relevant_docs" in state_updates else str(result)[:200]
                )
            })
            outputs.append(
                ToolMessage(
                    content=json.dumps(result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        #print(f"[DEBUG]: returning after toolcall with ouputs: {outputs}\n\nand updated state: {updated_state}")
        return Command(update={**updated_state, "messages": outputs})


# ============================================================
# === Conditional Routing
# ============================================================
def route_tools(state: AgenticRAGState):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("No messages found in state")
    
    if len(state.get("tool_log", [])) > 10:
        print("[STOP] Too many tool calls; ending early.")
        return END
    
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    return END


# ============================================================
# === Agentic LangGraph
# ============================================================
class AgenticLangGraph:
    """
    Encapsulates the setup and execution of the Agentic LangGraph pipeline.
    Provides a `.run(query)` method that returns the generated answer,
    relevant docs/nodes, and tool call logs.
    """

    def __init__(self, model_name: str = "gpt-oss:20b"):
        self.tool_log = []
        self.graph = None
        self.model_name = model_name
        self.memory = InMemorySaver()
        self.get_node_info_tool = None
        
    async def reset_graph(self):
        """
        Public method: safely rebuild the LangGraph instance and free memory.
        Can be called by evaluators or orchestrators between long runs.
        """
        import gc, torch

        print("[RESET] Rebuilding Agentic LangGraph and clearing memory...")

        try:
            # Step 1: Explicitly delete large objects
            if hasattr(self, "graph"):
                del self.graph
            if hasattr(self, "memory"):
                del self.memory

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step 2: Recreate clean memory and rebuild the graph
            from langgraph.checkpoint.memory import InMemorySaver
            self.memory = InMemorySaver()
            await self.setup_graph()

            print("[RESET] Graph successfully rebuilt.")
        except Exception as e:
            print(f"[WARN] Graph reset failed: {e}")
            

    async def setup_graph(self):
        print("[INIT] Connecting to MCP servers...")

        # Connect MCP servers
        client = MultiServerMCPClient(
            {
                "qdrant": {
                    "url": "http://localhost:8766/mcp",
                    "transport": "streamable_http",
                },
                "neo4j": {
                    "url": "http://localhost:8765/mcp",
                    "transport": "streamable_http",
                },
            }
        )

        tools = await client.get_tools()
        print(f"[INIT] Loaded MCP tools: {[t.name for t in tools]}")

        # Add filter builder tool
        filter_builder_tool = StructuredTool.from_function(
                coroutine=build_qdrant_metadata_type_filter,
                name="filter_builder",
                description=(
                    "Builds a Qdrant filter (qdrant_client.models.Filter) from a natural-language description. "
                    "Supports inclusive matching (e.g., 'functions and issues' → multiple types)."
                ),
                args_schema=FilterBuilderInput,
            )
        tools.append(filter_builder_tool)

        # Wrap qdrant_search
        wrapped_tools = []
        for t in tools:
            if t.name == "qdrant_search":
                wrapped_tools.append(QdrantSearchWrapper(t))
            elif t.name == "get_node_info":
                async def get_node_info_wrapper(node_id: str, field: str, _tool=t):
                    # Force field to string in case model sends int
                    field = str(field)
                    return await _tool.ainvoke({"node_id": node_id, "field": field})

                wrapped_tool = StructuredTool.from_function(
                    coroutine=get_node_info_wrapper,
                    name=t.name,
                    description="Retrieve a specific property field from a Neo4j node (by global_id).",
                    args_schema=GetNodeInfoInput,
                )
                wrapped_tools.append(wrapped_tool)
                self.get_node_info_tool = wrapped_tool
            else:
                wrapped_tools.append(t)

        basic_tool_node = BasicToolNode(wrapped_tools)

        print(f"Building model with: {self.model_name}")
        llm = ChatOllama(
            model=self.model_name,
            base_url="http://localhost:11434",
        )
        llm_with_tools = llm.bind_tools(tools)

        async def llm_call(state: AgenticRAGState):
            # --- Build tool descriptions ---
            available_tools = [
                f"- **{t.name}**: {t.description.strip()}"
                for t in tools  # captured from outer scope
            ]
            tools_list_text = "\n".join(available_tools)

            # --- Compose system prompt ---
            system_prompt = f"""You are an **Agentic RAG assistant** designed to answer technical questions
        about a software codebase, repository, or knowledge graph.

        Your capabilities:
        - You can use external tools via function calls to look up information.
        - When retrieving documents, prefer the most semantically relevant and cite them if useful.
        - Summarize results clearly and concisely in Markdown.
        - When reasoning, explain key findings briefly before giving the final answer.

        Available tools:
        {tools_list_text}

        You should decide when to call tools, combine retrieved data, and then produce a final textual answer.
        """

            # --- Optionally, include retrieved context ---
            docs_text = "\n\n".join(d.page_content for d in state.get("relevant_docs", [])[:5])
            if docs_text:
                system_prompt += f"\n\nContext documents:\n{docs_text}"

            # --- Prepend system message ---
            messages = [{"role": "system", "content": system_prompt}] + state["messages"]

            response = await llm_with_tools.ainvoke(messages)
            return Command(update={"messages": [response]})


        graph_builder = StateGraph(AgenticRAGState)
        graph_builder.add_node("chatbot", llm_call)
        graph_builder.add_node("tools", basic_tool_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})

        self.graph = graph_builder.compile(checkpointer=self.memory)
        print("[INIT] Graph compiled successfully.")
        print(self.graph.get_graph().draw_ascii())

    async def _enrich_functions_from_docs(self, docs: List[Document]) -> List[str]:
        """Helper to get function names from docs (logic from V2)."""
        if not self.get_node_info_tool:
            print("[WARN] get_node_info_tool not available. Cannot enrich functions.")
            return []

        async def get_function_name(global_id: str):
            """Async helper to safely call the tool."""
            try:
                # Call the tool to get the combinedName
                name = await self.get_node_info_tool.ainvoke({
                    "node_id": global_id,
                    "field": "combinedName"
                })
                return name.strip() if name else None
            except Exception as e:
                print(f"[WARN] get_node_info failed for {global_id}: {e}")
                return None
        
        function_tasks = []

        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            if not isinstance(meta, dict):
                continue
            doc_type = meta.get("type", "").lower()
            raw_id = meta.get("node_id")

            # 2. If it's a function, create a task to get its name
            if "function" in doc_type and raw_id:
                try:
                    node_prefix = doc_type.split('_')[0].upper()
                    global_id = f"{node_prefix}:{raw_id}"
                    function_tasks.append(get_function_name(global_id))
                except Exception:
                    continue

        # Run all function name lookups in parallel
        function_names = await asyncio.gather(*function_tasks)
        function_names = [f for f in function_names if isinstance(f, str)]
        
        # Filter out None values and duplicates
        return sorted(list(set(name for name in function_names if name)))
    
    async def run_async(self, query: str,session_id:str=None) -> Dict[str, Any]:
        """Run the LangGraph pipeline for one query asynchronously."""
        if not self.graph:
            await self.setup_graph()

        final_state = None
        if not session_id:
            session_id = str(uuid.uuid4())
        #print(f"[DEBUG] Running with session ID: {session_id}")

        async for event in self.graph.astream_events(
            {
                "query": query,
                "messages": [{"role": "user", "content": query}],
                "relevant_docs": [],
                "relevant_functions": [],
                "relevant_node_ids": [],
                "tool_log": [],
            },
            config={"configurable": {"thread_id": session_id}},
            version="v1",
            stream_mode="values",
        ):
            if event["event"] == "on_value":
                # Each emitted merged state snapshot
                state = event["data"]
                #print(f"[DEBUG] on_value — merged state keys: {list(state.keys())}")
                final_state = state
            elif event["event"] == "on_chain_end":
                # Backup — some LangGraph versions emit this too
                if "data" in event and "output" in event["data"]:
                    final_state = event["data"]["output"]

        if not final_state:
            raise RuntimeError("Graph did not produce a final state (no on_value or on_chain_end).")

        messages = final_state.get("messages", [])
        answer = ""
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                answer = last.content
            elif isinstance(last, dict) and "content" in last:
                answer = last["content"]
        docs = final_state.get("relevant_docs", [])
        serializable_docs = [
            {"page_content": d.page_content, "metadata": d.metadata} for d in docs
        ]
        node_ids = final_state.get("relevant_node_ids", [])
        tool_log = final_state.get("tool_log", [])

        functions = await self._enrich_functions_from_docs(docs)

        return {
            "answer": answer,
            "relevant_docs": serializable_docs,
            "relevant_functions": functions,
            "relevant_node_ids": node_ids,
            "tool_log": tool_log,
        }

    def run(self, query: str, session_id:str=None) -> Dict[str, Any]:
        """Synchronous wrapper for compatibility with evaluators."""
        return asyncio.run(self.run_async(query, session_id))

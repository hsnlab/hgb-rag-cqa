import asyncio
import json
import os
import torch
import gc
import httpx
import uuid
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.graph import add_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from qdrant_client import models
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


def merge_dict_of_lists(left, right):
    """Safely merge dicts whose values are lists or sets."""
    if right == "CLEAR":
        return {}
    if left is None:
        left = {}
    if right is None:
        right = {}
    merged = {k: set(v) for k, v in left.items()}
    for k, v in right.items():
        merged.setdefault(k, set()).update(v)
    return {k: sorted(list(v)) for k, v in merged.items()}


class AgenticRAGState(TypedDict):
    query: str
    relevant_node_ids: Annotated[List[str], add_list]
    relevant_functions: Annotated[List[str], add_list]
    relevant_docs: Annotated[List[Document], add_list]
    context_graph: Annotated[List[dict], add_list]
    messages: Annotated[list, add_messages]
    tool_log: Annotated[List[dict], add_list]


# ============================================================
# === Qdrant Filter Builder (optional)
# ============================================================
async def build_qdrant_metadata_type_filter(query: str) -> models.Filter:
    """Simple heuristic filter builder."""
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
        return models.Filter(must=[]).model_dump()

    condition = models.FieldCondition(
        key="metadata.type",
        match=models.MatchAny(any=matched_types),
    )
    return models.Filter(must=[condition]).model_dump()


# ============================================================
# === Wrapper: Qdrant Search Tool
# ============================================================
class QdrantSearchWrapper:
    def __init__(self, tool):
        self.tool = tool

    async def ainvoke(self, args, **kwargs):
        try:
            result = await self.tool.ainvoke(args, **kwargs)
            if asyncio.iscoroutine(result):
                print("[WARN] qdrant_search returned coroutine — awaiting it.")
                result = await result
        except Exception as e:
            print(f"[ERROR] qdrant_search failed: {e}")
            return [], {}

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = []

        docs = [Document(page_content=r["content"], metadata=r["metadata"]) for r in result]

        enriched_state = {
            "relevant_docs": docs,
            "tool_log": [{"tool": "qdrant_search", "args": args}],
        }
        return result, enriched_state


# ============================================================
# === Node: Qdrant Retrieval (with optional filter builder)
# ============================================================
class RetrievalQuery(BaseModel):
    """The optimized query or queries for vector search."""
    search_query: str = Field(
        ...,
        description="The primary, optimized, technical search query (no conversational filler) derived from the user's request. This must be the best single query for vector search."
    )
    # If you wanted multi-query retrieval, you would add a list field here:
    # secondary_queries: List[str] = Field(..., description="A list of 1-3 additional queries for complex concepts.")


# ============================================================
# === Node: Qdrant Retrieval (with Query Rewriting)
# ============================================================
class QdrantRetrievalNode:
    def __init__(self, qdrant_tool, llm, filter_builder_tool=None):
        self.qdrant_tool = qdrant_tool
        self.filter_builder_tool = filter_builder_tool
        
        self.query_rewriter_llm = llm.with_structured_output(
            schema=RetrievalQuery,
        )


    async def __call__(self, state: AgenticRAGState):
        original_query = state["query"]
        
        rewritten_query = original_query
        
        try:
            prompt = f"""You are a Query Optimizer. Your job is to transform a conversational user query into a single, highly optimized, technical query that is best suited for dense vector search (semantic similarity).
            
            Focus on key entities, function names, class names, and concepts. Remove all conversational fluff ("please," "can you tell me," "I need").
            
            Original Query: "{original_query}"
            
            Generate the best possible vector search query using the provided JSON schema."""
            
            # Note: We use the raw LLM with structured output binding
            response: RetrievalQuery = await self.query_rewriter_llm.ainvoke(
                [HumanMessage(content=prompt)]
            )
            
            rewritten_query = response.search_query
            # print(f"[DEBUG] Query Rewritten:\n  Original: {original_query}\n  Optimized: {rewritten_query}")

        except Exception as e:
            print(f"[WARN] Query rewriting failed: {e}. Using original query.")
            rewritten_query = original_query
        
       
        search_query = rewritten_query
        user_filter = state.get("metadata_filter", None)

        if user_filter:
            metadata_filter = user_filter
        elif self.filter_builder_tool:
            try:
               metadata_filter = await self.filter_builder_tool.ainvoke({"query": original_query})
            except Exception as e:
                print(f"[WARN] Filter builder failed, proceeding without filter: {e}")
                metadata_filter = {}
        else:
            metadata_filter = {}

        # --- 2. Perform Retrieval with the Optimized Query ---
        args = {"query": search_query, "metadata_filter": metadata_filter}
        result, enriched = await self.qdrant_tool.ainvoke(args)
        
        # We should also log the optimized query for tracking/debugging
        enriched["tool_log"].append({
            "tool": "Query Rewriter",
            "args": {"original_query": original_query},
            "result_summary": f"Optimized query used: {search_query[:80]}..."
        })
        
        return Command(update=enriched)

class AgenticRetrievalNode:
    """
    A more flexible retrieval node that allows the LLM to reason and decide
    how to query Qdrant multiple times before returning results.
    """

    def __init__(self, qdrant_tool, llm, max_rounds: int = 3):
        self.qdrant_tool = qdrant_tool
        self.llm = llm
        self.max_rounds = max_rounds

    async def __call__(self, state: AgenticRAGState):
        query = state["query"]
        messages = [
            {"role": "system", "content": """You are an expert retrieval agent.
You have access to a Qdrant search tool.
Your goal: find the most relevant technical documents for the user's query.
You may issue multiple searches to refine your results.
When confident, summarize what you found and return the documents.
Use the 'qdrant_search' tool ONLY for searching.
"""},
            {"role": "user", "content": query},
        ]

        # Define a loop: up to N reasoning + tool steps
        all_docs = []
        for step in range(self.max_rounds):
            try:
                response = await self.llm.ainvoke(messages)
            except Exception as e:
                print(f"[WARN] LLM retrieval reasoning failed: {e}")
                break

            # Parse the LLM response — it might "propose" a query
            if "search" in response.content.lower():
                search_query = response.content.split("search", 1)[1].strip(": ").split("\n")[0]
                print(f"[AGENTIC-RETRIEVAL] Step {step+1}: Searching Qdrant for '{search_query}'")

                try:
                    docs_result = await self.qdrant_tool.ainvoke({"query": search_query})
                except Exception as e:
                    print(f"[ERROR] Qdrant search failed: {e}")
                    continue

                if isinstance(docs_result, str):
                    try:
                        docs_result = json.loads(docs_result)
                    except json.JSONDecodeError:
                        docs_result = []

                docs = [
                    Document(page_content=d["content"], metadata=d["metadata"])
                    for d in docs_result
                ]
                all_docs.extend(docs)

                # Feed back docs into LLM reasoning loop
                sample_text = "\n\n".join(d.page_content[:300] for d in docs[:3])
                messages.append({
                    "role": "system",
                    "content": f"Retrieved documents:\n{sample_text}\n\nThink about whether to refine or finalize."
                })
                continue
            else:
                # LLM indicates it’s done
                print("[AGENTIC-RETRIEVAL] LLM decided to stop searching.")
                break

        if not all_docs:
            return Command(update={"relevant_docs": [], "tool_log": [{"tool": "qdrant_search", "args": {"query": query}}]})

        return Command(update={
            "relevant_docs": all_docs,
            "tool_log": [{"tool": "agentic_retrieval", "args": {"query": query}, "rounds": len(all_docs)}],
        })

# ============================================================
# === Node: Relevancy Checker
# ============================================================
class IndicesToKeep(BaseModel):
    """List of indices of documents that are relevant to the query."""
    indices: List[int] = Field(
        description="A list of integer indices (e.g., [0, 2, 5]) corresponding to the relevant documents."
    )

class RelevancyCheckerNode:
    def __init__(self, llm):
        self.structured_llm = llm.with_structured_output(schema=IndicesToKeep)

    async def __call__(self, state: AgenticRAGState):
        query = state["query"]
        docs = state.get("relevant_docs", [])
        if not docs:
            return Command(update={})

        docs_text = "\n\n".join(f"Doc {i}: {d.page_content}" for i, d in enumerate(docs[:10]))
        system_prompt = """You are a strict relevance filter.
        Given the query and retrieved docs, remove any docs that are not clearly relevant.
        Respond with a JSON list of indices to keep. """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nDocuments:\n{docs_text}"},
        ]

        try:
            response = await self.structured_llm.ainvoke(messages)
            indices = response.indices
            filtered_docs = [docs[i] for i in indices if 0 <= i < len(docs)]
        except Exception as e:
            print(f"[WARN] relevancy_checker failed: {e}")
            filtered_docs = docs

        return Command(update={"relevant_docs": filtered_docs})


# ============================================================
# === Node: Neo4j Enrichment (structured node dict)
# ============================================================
class NodeEnrichmentNode:
    """Enrich retrieved docs with structured Neo4j info using get_node_info."""

    def __init__(self, get_node_info_tool):
        self.get_node_info_tool = get_node_info_tool

    async def __call__(self, state: AgenticRAGState) -> Command:
        docs = state.get("relevant_docs", [])
        if not docs:
            return Command(update={})

        async def get_function_name(global_id: str):
            """Async helper to safely call the tool."""
            try:
                # Call the tool to get the combinedName
                name = await self.get_node_info_tool.ainvoke({
                    "node_id": global_id,
                    "field": "combinedName" # Requesting combinedName
                })
                return name.strip() if name else None
            except Exception as e:
                print(f"[WARN] get_node_info failed for {global_id}: {e}")
                return None
        
        global_ids = set()
        function_tasks = []

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
                global_ids.add(global_id)
            except Exception:
                print(f"[WARN] Could not parse global_id from doc_type: {doc_type}")
                continue

            # 2. If it's a function, create a task to get its name
            if "function" in doc_type:
                function_tasks.append(get_function_name(global_id))

        # Run all function name lookups in parallel
        function_names = await asyncio.gather(*function_tasks)
        
        # Filter out None values from failed lookups
        valid_function_names = [name for name in function_names if name]

        return Command(update={
            "relevant_node_ids": sorted(list(global_ids)),
            "relevant_functions": sorted(list(valid_function_names))
        })


# ============================================================
# === Node: Graph Context Builder (Neo4j MST + Context)
# ============================================================
class GraphContextBuilderNode:
    def __init__(self, mst_tool, edge_context_tool):
        self.mst_tool = mst_tool
        self.edge_context_tool = edge_context_tool

    async def __call__(self, state: AgenticRAGState):
        node_ids_list = state.get("relevant_node_ids", [])
        node_ids = list(set(node_ids_list))

        if not node_ids:
            return Command(update={"graph_context_text": "No node IDs found."})

        mst_result = await self.mst_tool.ainvoke({"node_ids": node_ids})
        try:
            edges = json.loads(mst_result) if isinstance(mst_result, str) else mst_result
        except Exception:
            edges = []

        if not edges:
            return Command(update={"graph_context_text": "No connections found among retrieved nodes."})

        records = await self.edge_context_tool.ainvoke({"edges": edges})
        try:
            records = json.loads(records) if isinstance(records, str) else records
        except Exception:
            records = []

        context_lines = ["### Neo4j Minimal Spanning Tree Context"]
        for r in records:
            s_lbl = ",".join(r.get("source_labels", []))
            t_lbl = ",".join(r.get("target_labels", []))
            rel = r.get("rel_type", "RELATED_TO")
            context_lines.append(f"- {r['source_id']} ({s_lbl}) -[{rel}]-> {r['target_id']} ({t_lbl})")

        context_text = "\n".join(context_lines)
        return Command(update={"graph_context_text": context_text, "context_graph": records})


# ============================================================
# === Node: Chatbot (LLM synthesis)
# ============================================================
def make_chatbot_node(llm):
    def chatbot_node(state: AgenticRAGState):
        docs_text = "\n\n".join(d.page_content for d in state.get("relevant_docs", [])[:5])
        graph_context = state.get("graph_context_text", "")
        system_prompt = f"""You are an Agentic RAG assistant.
Use both semantic documents and graph context to answer clearly.

Documents:
{docs_text}

Graph Context:
{graph_context}
"""
        messages = [{"role": "system", "content": system_prompt}] + state.get("messages", [])
        response = llm.invoke(messages)
        return Command(update={"messages": [response]})

    return chatbot_node

# ============================================================
# === Node: Query Planner / Router
# ============================================================
class NextAction(BaseModel):
    """The action the RAG pipeline should take next."""
    action: str = Field(
        ...,
        description="The next action to take. Must be 'rerun_retrieval' to perform another search, or 'proceed_to_synthesis' to generate the final answer."
    )

class QueryPlannerNode:
    def __init__(self, llm):
        self.structured_llm = llm.with_structured_output(schema=NextAction)

    async def __call__(self, state: AgenticRAGState) -> str:
        query = state["query"]
        docs = state.get("relevant_docs", [])
        
        # Determine the next action based on LLM instruction
        prompt = f"""You are a pipeline coordinator. Analyze the current state and decide the next step.
        
        Original Query: {query}
        Documents Retrieved So Far: {len(docs)}
        
        If you have enough, highly relevant documents (e.g., more than 3) to fully answer the query, choose 'proceed_to_synthesis'.
        If you have very few documents (e.g., 3 or less) or the existing documents seem insufficient or tangential, choose 'rerun_retrieval'.
        
        Note: You cannot modify the query in this step; only decide on the next action.
        """
        
        try:
            response: NextAction = await self.structured_llm.ainvoke(
                [{"role": "user", "content": prompt}]
            )
            if response.action not in ["proceed_to_synthesis", "rerun_retrieval"]:
                return "proceed_to_synthesis"
            else:
                return response.action
        except Exception as e:
            print(f"[ERROR] Query planner failed: {e}. Defaulting to synthesis.")
            # Fallback is crucial: always exit the loop if the LLM fails
            return "proceed_to_synthesis"

# ============================================================
# === Agentic LangGraph Setup
# ============================================================
class StrictAgenticLangGraph:
    def __init__(self, model_name: str = "mistral:7b"):
        self.model_name = model_name
        self.memory = InMemorySaver()
        self.graph = None
        
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

        client = MultiServerMCPClient(
            {
                "qdrant": {"url": "http://localhost:8766/mcp", "transport": "streamable_http"},
                "neo4j": {"url": "http://localhost:8765/mcp", "transport": "streamable_http"},
            }
        )

        tools = await client.get_tools()
        print(f"[INIT] Loaded MCP tools: {[t.name for t in tools]}")

        qdrant_tool = next(t for t in tools if t.name == "qdrant_search")
        mst_tool = next(t for t in tools if t.name == "minimal_spanning_tree")
        edge_context_tool = next(t for t in tools if t.name == "get_edge_context")
        get_node_info_tool = next(t for t in tools if t.name == "get_node_info")
        filter_builder_tool = next((t for t in tools if t.name == "filter_builder"), None)

        wrapped_qdrant = QdrantSearchWrapper(qdrant_tool)
        os.environ["OLLAMA_KEEP_ALIVE"] = "0"
        llm = ChatOllama(
            model=self.model_name, 
            base_url="http://localhost:11434", 
            async_client_kwargs={
                "headers": {"Connection": "close"},
                "timeout": 120,
                "limits": httpx.Limits(
                    max_keepalive_connections=0, 
                    max_connections=10,           
                ),
            },
        )

        retrieval_node = QdrantRetrievalNode(wrapped_qdrant, llm, filter_builder_tool)
        relevancy_node = RelevancyCheckerNode(llm)
        enrichment_node = NodeEnrichmentNode(get_node_info_tool)
        context_builder_node = GraphContextBuilderNode(mst_tool, edge_context_tool)
        query_planner_node = QueryPlannerNode(llm)
        chatbot_node = make_chatbot_node(llm)

        graph_builder = StateGraph(AgenticRAGState)
        graph_builder.add_node("retrieval", retrieval_node)
        graph_builder.add_node("relevancy", relevancy_node)
        graph_builder.add_node("enrichment", enrichment_node)
        graph_builder.add_node("context_builder", context_builder_node)
        graph_builder.add_node("query_planner", query_planner_node)
        graph_builder.add_node("chatbot", chatbot_node)

        graph_builder.add_edge(START, "retrieval")
        graph_builder.add_edge("retrieval", "relevancy")
        graph_builder.add_edge("relevancy", "enrichment")
        graph_builder.add_edge("enrichment", "context_builder")
        #graph_builder.add_edge("context_builder", "query_planner")
        graph_builder.add_conditional_edges(
            "context_builder", 
            query_planner_node,
            {
                "rerun_retrieval": "retrieval",        
                "proceed_to_synthesis": "chatbot",     
            }
        )
        graph_builder.add_edge("chatbot", END)

        self.graph = graph_builder.compile(checkpointer=self.memory)
        print("[INIT] Graph compiled successfully.")
        print(self.graph.get_graph().draw_ascii())

    async def run_async(self, query: str,session_id:str=None) -> Dict[str, Any]:
        """Run the LangGraph pipeline for one query asynchronously."""
        if not self.graph:
            await self.setup_graph()

        final_state = None
        if not session_id:
            session_id = str(uuid.uuid4())
        try:
            async for event in self.graph.astream_events(
                {
                    "query": query,
                    "messages": [{"role": "user", "content": query}],
                    "relevant_docs": [],
                    "relevant_node_ids": [],
                    "relevant_functions":[],
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
        except Exception as e:
            print(f"[ERROR] Graph streaming failed: {e}")
            traceback.print_exc()
            return {
                "answer": None,
                "relevant_docs": [],
                "relevant_node_ids": [],
                "relevant_functions": [],
                "tool_log": [{"error": str(e)}],
            }
        if not final_state:
            raise RuntimeError("Graph did not produce a final state (no on_value or on_chain_end).")

        messages = final_state.get("messages", [])
        answer = messages[-1].content if messages else None
        docs = final_state.get("relevant_docs", [])
        serializable_docs = [
            {"page_content": d.page_content, "metadata": d.metadata} for d in docs
        ]
        node_ids = final_state.get("relevant_node_ids", [])
        functions = final_state.get("relevant_functions",[])
        tool_log = final_state.get("tool_log", [])
        
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "answer": answer,
            "relevant_docs": serializable_docs,
            "relevant_node_ids": node_ids,
            "relevant_functions":functions,
            "tool_log": tool_log,
        }

    def run(self, query: str, session_id:str=None) -> Dict[str, Any]:
        """Synchronous wrapper for compatibility with evaluators."""
        return asyncio.run(self.run_async(query, session_id))


# ============================================================
# === Run Example
# ============================================================
if __name__ == "__main__":
    agent = StrictAgenticLangGraph(model_name="gpt-oss:20b")
    result = agent.run("What pca variants are there in this repo?")
    print(json.dumps(result, indent=2))

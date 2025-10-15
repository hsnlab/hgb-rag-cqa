import asyncio
from dataclasses import Field
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
import textwrap
import traceback
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional, Annotated, Tuple
from langchain_core.documents import Document
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langchain.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import models
from pydantic import BaseModel, Field
import json


class AgenticRAGState(TypedDict):
    """
    State object that flows through the LangGraph.
    """
    query: str
    relevant_node_ids: List[str]
    relevant_docs: List[Document]


    messages: Annotated[list, add_messages]

class QdrantSearchInput(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[dict] = None

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
        print(f"[DEBUG] Calling wrapped qdrant_search with args={args}")
        result = await self.tool.ainvoke(args, **kwargs)

        if isinstance(result, str):
            try:
                result = json.loads(result)
                print(f"[DEBUG] Parsed JSON string into {len(result)} results.")
            except json.JSONDecodeError:
                print("[WARN] Could not parse qdrant_search result as JSON.")
                result = []

        if not isinstance(result, list):
            print(f"[WARN] Unexpected qdrant_search return type: {type(result)}")
            result = []
        # Convert raw dicts → LangChain Documents
        docs = [Document(page_content=r["content"], metadata=r["metadata"]) for r in result]

        # Extract node IDs (adjust key names as needed)
        node_ids = []
        for doc in docs:
            mid = doc.metadata.get("node_id")
            if mid:
                node_ids.append(str(mid))

        enriched_state = {
            "relevant_docs": docs,
            "relevant_node_ids": node_ids,
        }

        print(f"[DEBUG] Wrapped qdrant_search returned {len(docs)} docs and {len(node_ids)} node IDs.")
        return result, enriched_state
async def qdrant_search_with_state(tool, query: str, top_k: int = 5, filters: Optional[dict] = None) -> Tuple[List[dict], dict]:
    """
    Run MCP qdrant_search and enrich the result with documents and node IDs for state.
    
    Returns:
        - Raw qdrant_search result (list of dicts)
        - Enriched state: {'relevant_docs': [...], 'relevant_node_ids': [...]}
    """
    print(f"[DEBUG] Calling qdrant_search with query={query}, top_k={top_k}, filters={filters}")
    # Call the underlying MCP tool
    result = await tool.ainvoke(query=query, top_k=top_k, filters=filters)

    # Parse JSON string if returned
    if isinstance(result, str):
        try:
            result = json.loads(result)
            print(f"[DEBUG] Parsed JSON string into {len(result)} results.")
        except json.JSONDecodeError:
            print("[WARN] Could not parse qdrant_search result as JSON.")
            result = []

    if not isinstance(result, list):
        print(f"[WARN] Unexpected qdrant_search return type: {type(result)}")
        result = []

    # Convert raw dicts → LangChain Documents
    docs = [Document(page_content=r["content"], metadata=r["metadata"]) for r in result]

    # Extract node IDs
    node_ids = [str(doc.metadata.get("node_id")) for doc in docs if doc.metadata.get("node_id")]

    enriched_state = {
        "relevant_docs": docs,
        "relevant_node_ids": node_ids,
    }

    print(f"[DEBUG] qdrant_search returned {len(docs)} docs and {len(node_ids)} node IDs.")
    return result, enriched_state

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

    print(f"[DEBUG] Built Qdrant filter for '{query}': {qdrant_filter}")

    return qdrant_filter.model_dump()

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
        updated_state = {}

        for tool_call in message.tool_calls:
            tool = self.tools_by_name[tool_call["name"]]

            tool_result = await tool.ainvoke(tool_call["args"])

            # Handle wrapped tools that return (result, state_updates)
            if isinstance(tool_result, tuple) and len(tool_result) == 2:
                result, state_updates = tool_result
                updated_state.update(state_updates)
            else:
                result = tool_result

            outputs.append(
                ToolMessage(
                    content=json.dumps(result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs, **updated_state}


def route_tools(state: AgenticRAGState):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages:= state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("No messages found in state")
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    else:
        return END



async def main():
    print("Connecting to MCP servers...")

    client = MultiServerMCPClient(
        {
            "qdrant": {
                "url": "http://localhost:8766/mcp",
                "transport": "streamable_http",
            },
            "neo4j": {
                "url": "http://localhost:8765/mcp",
                "transport": "streamable_http",
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded MCP tools: {[t.name for t in tools]}")
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
    wrapped_tools = []
    for t in tools:
        if t.name == "qdrant_search":
            #wrapped_tools.append(
            #    StructuredTool.from_function(
            #        coroutine = lambda query, top_k=5, filters=None: qdrant_search_with_state(t, query, top_k, filters),
            #        name = t.name,
            #        description = t.description,
            #        args_schema = QdrantSearchInput
            #    )
            #)
            wrapped_tools.append(QdrantSearchWrapper(t))
        else:
            wrapped_tools.append(t)

    basic_tool_node = BasicToolNode(wrapped_tools)

    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0.2,
        base_url="http://localhost:11434",
    )
    
    #agent = create_react_agent(llm, tools)
    llm_with_tools = llm.bind_tools(tools)

    def llm_call(state: AgenticRAGState) -> str:
        """
        Call the LLM with the current state messages.
        """
        
        return {
            "messages": [ llm_with_tools.invoke(state["messages"])]
                
        }
    
    graph_builder = StateGraph(AgenticRAGState)
    graph_builder.add_node("chatbot", llm_call)
    graph_builder.add_node("tools", basic_tool_node)

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", END: END},
    )
    memory = InMemorySaver()
    graph = graph_builder.compile(checkpointer = memory)
    print("Agent ready. Type your query (or 'exit' to quit)\n")
    print(graph.get_graph().draw_ascii())

    async def stream_graph_updates(user_input: str):
        async for event in graph.astream({"messages": [{"role": "user", "content": user_input}]},config={"configurable": {"thread_id": "session_1"}}):
            for value in event.values():
                messages = value["messages"]
                last_msg = messages[-1]

                # Case 1: Assistant message (LLM response)
                if isinstance(last_msg, AIMessage):
                    #print(f"\nAssistant: {last_msg.content}\n")
                    last_msg.pretty_print()

                # Case 2: Tool message (result from MCP tool)
                elif isinstance(last_msg, ToolMessage):
                    last_msg.pretty_print()
                    #try:
                    #    result = json.loads(last_msg.content)
                    #except json.JSONDecodeError:
                    #    result = last_msg.content
#
                    #print(f"[DEBUG] Tool '{last_msg.name}' finished:")
                    #if isinstance(result, dict):
                    #    for key, val in result.items():
                    #        if isinstance(val, list):
                    #            print(f"  - {key}: list[{len(val)}]")
                    #        else:
                    #            short = str(val)
                    #            if len(short) > 120:
                    #                short = short[:117] + "..."
                    #            print(f"  - {key}: {short}")
                    #else:
                    #    short = str(result)
                    #    if len(short) > 200:
                    #        short = short[:197] + "..."
                    #    print(f"  {short}")
                else:
                    print(f"[DEBUG] Message type {type(last_msg)} -> {last_msg}")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            await stream_graph_updates(user_input)
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()
            break

if __name__ == "__main__":
    asyncio.run(main())

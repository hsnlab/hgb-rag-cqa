import asyncio
import json
import traceback
import httpx
import torch
import gc
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langchain.tools import StructuredTool
from langgraph.types import Command
from langgraph.graph import add_messages
from qdrant_client import models
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated


# ============================================================
# === State & Merge Helpers
# ============================================================
def add_int(left, right):
    if right == "CLEAR":
        return 0
    if left is None:
        left = 0
    if right is None:
        right = 0
    return left + right

def add_list(left, right):
    """Safe list concatenation helper for LangGraph state merges."""
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
    """Base state shared between Free and Strict LangGraph workflows."""
    query: str
    relevant_node_ids: Annotated[List[str], add_list]
    relevant_functions: Annotated[List[str], add_list]
    relevant_docs: Annotated[List[Document], add_list]
    messages: Annotated[list, add_messages]
    retrieval_attempts: Annotated[int, add_int]
    tool_log: Annotated[List[dict], add_list]


# ============================================================
# === MCP Tool Utilities
# ============================================================

def get_tool(tools: list, name: str):
    """Safely retrieve a tool by name, raising a clear error if missing."""
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        raise RuntimeError(
            f"[FATAL] Required tool '{name}' not found among available MCP tools: "
            f"{[t.name for t in tools]}"
        )
    return tool

# ============================================================
# === Node inof input schema
# ============================================================

class GetNodeInfoInput(BaseModel):
    node_id: str = Field(..., description="The global node ID, e.g. FUNC:12345")
    field: str = Field(..., description="The property name to retrieve (must be a string).")

# ============================================================
# === Filter Builder
# ============================================================

class FilterBuilderInput(BaseModel):
    query: str = Field(..., description="Natural language description of what to filter for")


async def build_qdrant_metadata_type_filter(query: str) -> models.Filter:
    """Build a Qdrant-compatible Filter object from a natural-language query."""
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
    return qdrant_filter.model_dump()


# ============================================================
# === Qdrant Wrapper
# ============================================================

class QdrantSearchWrapper:
    """Wraps an MCP qdrant_search tool to produce structured LangChain Documents."""

    def __init__(self, tool):
        self.tool = tool
        self.name = tool.name
        self.description = getattr(tool, "description", "")
        self.args_schema = getattr(tool, "args_schema", None)

    async def ainvoke(self, args, **kwargs):
        """Run the wrapped MCP tool asynchronously and enrich its result."""
        print(f"[DEBUG] Calling wrapped qdrant_search with args={args}")
        try:
            result = await self.tool.ainvoke(args, **kwargs)
        except Exception as e:
            print(f"[ERROR] qdrant_search failed: {e}")
            print("[FALLBACK] Retrying with empty metadata_filter...")

            # Retry with safe fallback filter
            fallback_args = dict(args)
            fallback_args["metadata_filter"] = {}
            try:
                result = await self.tool.ainvoke(fallback_args, **kwargs)
                print("[FALLBACK] Retry succeeded.")
            except Exception as e2:
                print(f"[ERROR] Fallback qdrant_search also failed: {e2}")
                return [], {}

        # Handle async coroutine returns
        if asyncio.iscoroutine(result):
            print("[WARN] qdrant_search returned coroutine — awaiting it.")
            result = await result

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                print("[WARN] Could not parse qdrant_search result as JSON.")
                result = []

        if not isinstance(result, list):
            print(f"[WARN] Unexpected qdrant_search return type: {type(result)}")
            result = []

        docs = [Document(page_content=r["content"], metadata=r["metadata"]) for r in result]

        node_ids = set()
        for doc in docs:
            meta = doc.metadata
            doc_type = meta.get("type", "").lower()
            raw_id = meta.get("node_id")

            if not doc_type or not raw_id:
                continue
            try:
                node_prefix = doc_type.split("_")[0].upper()
                node_ids.add(f"{node_prefix}:{raw_id}")
            except Exception:
                print(f"[WARN] Could not parse global_id from doc_type: {doc_type}")

        enriched_state = {
            "relevant_docs": docs,
            "relevant_node_ids": sorted(list(node_ids)),
        }

        print(f"[DEBUG] Wrapped qdrant_search returned {len(docs)} docs and {len(node_ids)} node IDs.")
        return result, enriched_state


# ============================================================
# === Basic Tool Node
# ============================================================

class BasicToolNode:
    """Runs the tools requested in the latest AIMessage with strong error handling."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    async def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        updated_state = {"tool_log": inputs.get("tool_log", []).copy()}

        for tool_call in getattr(message, "tool_calls", []):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name not in self.tools_by_name:
                warn_msg = f"[WARN] LLM requested unknown tool '{tool_name}'. Skipping this call."
                print(warn_msg)
                updated_state["tool_log"].append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_summary": "Invalid tool name",
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
                    "result_summary": f"Error: {e}",
                })
                continue

            # Handle wrapped tools returning (result, state_updates)
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
                ),
            })

            outputs.append(
                ToolMessage(
                    content=json.dumps(result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )

        return Command(update={**updated_state, "messages": outputs})


# ============================================================
# === Conditional Routing
# ============================================================

def route_tools(state: AgenticRAGState):
    """Router deciding whether to call a tool node or end the workflow."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("No messages found in state")

#    if len(state.get("tool_log", [])) > 15:
#        print("[STOP] Too many tool calls; ending early.")
#        return "END"

    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    return "END"


# ============================================================
# === Utility: LLM Factory
# ============================================================

def make_llm(model_name: str, base_url: str = "http://localhost:11434", timeout: int = 120):
    """Factory for ChatOllama with consistent async and timeout configuration."""
    os_env = {"OLLAMA_KEEP_ALIVE": "0"}
    for k, v in os_env.items():
        import os
        os.environ[k] = v

    return ChatOllama(
        model=model_name,
        base_url=base_url,
        async_client_kwargs={
            "headers": {"Connection": "close"},
            "timeout": timeout,
            "limits": httpx.Limits(max_keepalive_connections=0, max_connections=10),
        },
    )


# ============================================================
# === Memory Cleanup Helper
# ============================================================

def cleanup_memory():
    """Free Python and CUDA memory between graph runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

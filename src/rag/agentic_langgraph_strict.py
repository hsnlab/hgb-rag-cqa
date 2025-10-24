import asyncio
import json
from typing import Dict, Any, List

from langchain_core.documents import Document
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
    relevant_node_ids: Annotated[Dict[str, List[str]], merge_dict_of_lists]
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
class QdrantRetrievalNode:
    def __init__(self, qdrant_tool, filter_builder_tool=None):
        self.qdrant_tool = qdrant_tool
        self.filter_builder_tool = filter_builder_tool

    async def __call__(self, state: AgenticRAGState):
        query = state["query"]
        # The agent or previous node may have provided a filter manually
        user_filter = state.get("metadata_filter", None)

        # Try to build one if none is provided and the builder exists
        if user_filter:
            metadata_filter = user_filter
        elif self.filter_builder_tool:
            try:
                metadata_filter = await self.filter_builder_tool.ainvoke({"query": query})
            except Exception as e:
                print(f"[WARN] Filter builder failed, proceeding without filter: {e}")
                metadata_filter = {}
        else:
            metadata_filter = {}

        args = {"query": query, "metadata_filter": metadata_filter}
        result, enriched = await self.qdrant_tool.ainvoke(args)
        return Command(update=enriched)


# ============================================================
# === Node: Relevancy Checker
# ============================================================
class RelevancyCheckerNode:
    def __init__(self, llm):
        self.llm = llm

    async def __call__(self, state: AgenticRAGState):
        query = state["query"]
        docs = state.get("relevant_docs", [])
        if not docs:
            return Command(update={})

        docs_text = "\n\n".join(f"Doc {i}: {d.page_content}" for i, d in enumerate(docs[:10]))
        system_prompt = """You are a strict relevance filter.
        Given the query and retrieved docs, remove any docs that are not clearly relevant.
        Respond with a JSON list of indices to keep."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nDocuments:\n{docs_text}"},
        ]

        try:
            response = await self.llm.ainvoke(messages)
            indices = json.loads(response.content)
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

    async def __call__(self, state: AgenticRAGState):
        docs = state.get("relevant_docs", [])
        if not docs:
            return Command(update={})
        enriched_ids = await self._enrich_async(docs)
        return Command(update={"relevant_node_ids": enriched_ids})

    async def _enrich_async(self, docs: List[Document]) -> Dict[str, List[str]]:
        function_names, issue_ids, pr_ids = set(), set(), set()

        async def get_info(node_id: str, field: str):
            try:
                result = await self.get_node_info_tool.ainvoke({"node_id": node_id, "field": field})
                if isinstance(result, str):
                    return result.strip()
                return str(result)
            except Exception as e:
                print(f"[DEBUG] get_node_info failed for {node_id}:{field} → {e}")
                return ""

        tasks, meta_index = [], []
        for doc in docs:
            meta = doc.metadata
            doc_type = meta.get("type", "").lower()
            raw_id = meta.get("node_id")
            if not raw_id:
                continue

            if "function" in doc_type:
                tasks.append(asyncio.create_task(get_info(f"{doc_type.upper()}:{raw_id}", "combinedName")))
                meta_index.append(("function",))
            elif "issue" in doc_type:
                tasks.append(asyncio.create_task(get_info(f"{doc_type.upper()}:{raw_id}", "ID")))
                meta_index.append(("issue",))
            elif "pr" in doc_type:
                tasks.append(asyncio.create_task(get_info(f"{doc_type.upper()}:{raw_id}", "ID")))
                meta_index.append(("pr",))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (mtype,), val in zip(meta_index, results):
            if not val or isinstance(val, Exception):
                continue
            if mtype == "function":
                function_names.add(val)
            elif mtype == "issue":
                issue_ids.add(val)
            elif mtype == "pr":
                pr_ids.add(val)

        return {
            "functions": sorted(function_names),
            "issues": sorted(issue_ids),
            "prs": sorted(pr_ids),
        }


# ============================================================
# === Node: Graph Context Builder (Neo4j MST + Context)
# ============================================================
class GraphContextBuilderNode:
    def __init__(self, mst_tool, edge_context_tool):
        self.mst_tool = mst_tool
        self.edge_context_tool = edge_context_tool

    async def __call__(self, state: AgenticRAGState):
        ids_map = state.get("relevant_node_ids", {})
        node_ids = []
        for k, v in ids_map.items():
            node_ids.extend(v)
        node_ids = list(set(node_ids))

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
# === Agentic LangGraph Setup
# ============================================================
class AgenticLangGraph:
    def __init__(self, model_name: str = "mistral:7b"):
        self.model_name = model_name
        self.memory = InMemorySaver()
        self.graph = None

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
        llm = ChatOllama(model=self.model_name, base_url="http://localhost:11434")

        retrieval_node = QdrantRetrievalNode(wrapped_qdrant, filter_builder_tool)
        relevancy_node = RelevancyCheckerNode(llm)
        enrichment_node = NodeEnrichmentNode(get_node_info_tool)
        context_builder_node = GraphContextBuilderNode(mst_tool, edge_context_tool)
        chatbot_node = make_chatbot_node(llm)

        graph_builder = StateGraph(AgenticRAGState)
        graph_builder.add_node("retrieval", retrieval_node)
        graph_builder.add_node("relevancy", relevancy_node)
        graph_builder.add_node("enrichment", enrichment_node)
        graph_builder.add_node("context_builder", context_builder_node)
        graph_builder.add_node("chatbot", chatbot_node)

        graph_builder.add_edge(START, "retrieval")
        graph_builder.add_edge("retrieval", "relevancy")
        graph_builder.add_edge("relevancy", "enrichment")
        graph_builder.add_edge("enrichment", "context_builder")
        graph_builder.add_edge("context_builder", "chatbot")
        graph_builder.add_edge("chatbot", END)

        self.graph = graph_builder.compile(checkpointer=self.memory)
        print("[INIT] Graph compiled successfully.")
        print(self.graph.get_graph().draw_ascii())

    async def run_async(self, query: str) -> Dict[str, Any]:
        if not self.graph:
            await self.setup_graph()

        final_state = None
        async for event in self.graph.astream_events(
            {
                "query": query,
                "messages": [{"role": "user", "content": query}],
                "relevant_docs": [],
                "relevant_node_ids": {},
                "context_graph": [],
                "tool_log": [],
            },
            config={"configurable": {"thread_id": "session_eval"}},
            version="v1",
            stream_mode="values",
        ):
            if event["event"] == "on_value":
                final_state = event["data"]

        if not final_state:
            raise RuntimeError("Graph did not produce a final state.")

        messages = final_state.get("messages", [])
        answer = messages[-1].content if messages else None

        return {
            "answer": answer,
            "relevant_docs": final_state.get("relevant_docs", []),
            "relevant_node_ids": final_state.get("relevant_node_ids", {}),
            "context_graph": final_state.get("context_graph", []),
            "graph_context_text": final_state.get("graph_context_text", ""),
            "tool_log": final_state.get("tool_log", []),
        }

    def run(self, query: str) -> Dict[str, Any]:
        return asyncio.run(self.run_async(query))


# ============================================================
# === Run Example
# ============================================================
if __name__ == "__main__":
    agent = AgenticLangGraph(model_name="mistral:7b")
    result = agent.run("How does pca.fit work?")
    print(json.dumps(result, indent=2))

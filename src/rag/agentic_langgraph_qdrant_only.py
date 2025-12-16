from src.rag.agentic_langgraph_v2 import AgenticLangGraph
from src.utils.langgraph_utils import get_tool

class QdrantOnlyAgenticLangGraph(AgenticLangGraph):
    """A minimal AgenticLangGraph restricted to Qdrant MCP tools."""

    async def collect_tools(self):
        _, wrapped_tools = await super().collect_tools()
        qdrant_tools = [
            t for t in wrapped_tools if "qdrant" in t.name or "filter_builder" in t.name
        ]
        return qdrant_tools, qdrant_tools
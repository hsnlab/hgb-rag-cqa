# retrieval_tool.py
import logging
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from src.utils.retrieval import KnowledgeGraphRetriever   # ⬅️ ITT történik az import


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


class RetrievalToolInput(BaseModel):
    query: str = Field(..., description="The question to retrieve technical context for.")


class RetrievalTool(BaseTool):
    name: str = "retrieval_tool"
    description: str = "Queries Neo4j and Qdrant for relevant code, issue and graph data."
    args_schema: Type[BaseModel] = RetrievalToolInput

    def __init__(self, retriever: KnowledgeGraphRetriever):
        super().__init__()
        self._retriever = retriever
        logging.info("🔧 RetrievalTool initialized with KnowledgeGraphRetriever.")

    def _run(self, query: str) -> str:
        RetrievalTool.call_count += 1
        logging.info(f"🔁 RetrievalTool call #{RetrievalTool.call_count}")
        logging.info(f"🔍 [RetrievalTool] Agent query: {query}")
        try:
            docs, qtype = self._retriever.retrieve(query, top_k=5)
            if not docs:
                logging.warning("⚠️ No documents retrieved.")
                return "No relevant documents found."

            logging.info(f"✅ Retrieved {len(docs)} documents (type={qtype}).")
            sample = "\n\n".join([d.page_content[:400] for d in docs[:3]])
            return f"Retrieved {len(docs)} documents (type={qtype}).\n\nSample:\n{sample}"

        except Exception as e:
            logging.error(f"❌ RetrievalTool error: {e}")
            return f"Retrieval error: {e}"

    async def _arun(self, query: str) -> str:
        """Optional async run (CrewAI supports async tool execution)"""
        return self._run(query)

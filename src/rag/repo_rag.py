from typing import List, Tuple, Dict, Any, Optional, Union
from langchain_core.documents import Document
from .simple_rag import SimpleRAG
from .config import PipelineConfig

class RepositoryRAG(SimpleRAG):
    """
    RepositoryRAG — repository-aware retrieval strategy.

    Minimal responsibilities:
      - in __init__: accept a KnowledgeGraphRetriever (self.retriever),
        optional deduplicator and reranker instances (wired but not
        enforced here).
      - implement _retrieve(query, config) -> (docs, query_type, scored_results)
        so the BaseRAG.run orchestration can dedup/rerank/generate.

    It intentionally *does not* reimplement enrichment or generation —
    those are inherited from SimpleRAG (so both RAGs output identical
    `top_nodes` and textual prompts).
    """

    def __init__(
        self,
        vectorstore,
        retriever,
        deduplicator=None,
        reranker=None,
        llm=None,
        llm_model: str = None,
        neo4j_uri: str = None,
        neo4j_auth: Union[Tuple[str, str], str, None] = None,
    ):
        """
        Args:
            vectorstore: backing vector store (Qdrant wrapper) — used as fallback
            retriever: KnowledgeGraphRetriever instance (must implement .retrieve(query, top_k))
            deduplicator: optional Deduplicator instance (provides .deduplicate(docs, ...))
            reranker: optional Reranker instance (provides .rerank(query, docs, ...))
            llm: (optional) pre-initialized HuggingFace pipeline or model wrapper
            llm_model: (optional) name of LLM to pass to SimpleRAG initializer
        """
        super().__init__(vectorstore=vectorstore, llm=llm, llm_model=llm_model,neo4j_auth=neo4j_auth, neo4j_uri=neo4j_uri)

        # KG-specific components
        self.retriever = retriever
        self.deduplicator = deduplicator
        self.reranker = reranker

    def _retrieve(self, query: str, config: PipelineConfig) -> Tuple[List[Document], str, Optional[List]]:
        """
        Repository retrieval.

        Returns a tuple: (docs, query_type)

        Behavior:
          - If config.retriever == "kg" (or retriever exists), use the KG retriever's logic.
          - Otherwise, fall back to SimpleRAG's vectorstore retrieval.

        Notes:
          - KnowledgeGraphRetriever.retrieve is expected to return:
              (docs, query_type)
        """
        # Prefer KG retrieval when configured and available
        use_kg = getattr(config, "retriever", "kg") == "kg"
        if use_kg and hasattr(self, "retriever") and self.retriever is not None:
            if getattr(config, "over_retrieve", False):
                effective_k = config.top_k * getattr(config, "over_retrieve_factor", 10)
            else:
                effective_k = config.top_k
            cap = getattr(config, "over_retrieve_cap", 200)
            effective_k = min(effective_k, cap)
            docs, query_type = self.retriever.retrieve(query, top_k=effective_k)
            return docs, query_type

        # Fallback: use SimpleRAG's vectorstore-based retrieval
        return super()._retrieve(query, config)

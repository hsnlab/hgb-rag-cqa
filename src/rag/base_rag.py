from typing import List, Dict, Any
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from .config import PipelineConfig

class BaseRAG(ABC):
    def __init__(self):
        pass

    def run(self, query: str, config: PipelineConfig) -> Dict[str, Any]:
        """
        Standard public method the evaluator calls.
        Returns dict with keys:
          - answer
          - top_docs
          - top_functions
          - query_type
          - scored_results (optional)
        """
        top_k = config.top_k
        # retrieval stage (implemented/overridden by subclass)
        retrieved, query_type = self._retrieve(query, config)
        if config.verbose:
            print(f"Retrieved {len(retrieved)} documents.")

        # optional deduplication
        if config.deduplicate and hasattr(self, "deduplicator"):
            retrieved = self.deduplicator.deduplicate(
                retrieved,
                use_minhash=config.dedup_use_minhash,
                jaccard_threshold=config.dedup_jaccard_threshold,
                use_semantic=config.dedup_use_semantic,
                sim_threshold=config.dedup_sim_threshold,
            )
            if config.verbose:
                print(f"Deduplicated to {len(retrieved)} documents.")


        # optional rerank
        if config.rerank and hasattr(self, "reranker"):
            reranked = self.reranker.rerank(
                query,
                retrieved,
                use_graph=config.rerank_use_graph,
                use_popularity=config.rerank_use_popularity,
            )
            # reranked is list[(doc,score)]
            # update retrieved list limited to top_k
            retrieved = [doc for doc, _ in reranked][:top_k]
        else:
            retrieved = retrieved[:top_k]    
        # enrichment (functions, issues, PRs)
        top_nodes = self._enrich(retrieved)

        # generation
        answer = self._generate_answer_from_docs(query, retrieved, config)

        return {
            "answer": answer,
            "top_docs": retrieved,
            "top_functions": top_nodes.get("functions", []),
            "top_nodes": top_nodes,
            "query_type": query_type or "default",
        }

    @abstractmethod
    def _retrieve(self, query: str, config: PipelineConfig):
        """Return (List[Document], query_type: str or None, scored_results: Optional[list])"""

    @abstractmethod
    def _enrich(self, docs: List[Document]):
        """Return dict of nodes: {
            "functions": [], 
            "issues": [], 
            "prs": []
        }"""

    @abstractmethod
    def _generate_answer_from_docs(self, query: str, docs: List[Document], config: PipelineConfig) -> str:
        """LLM generation"""

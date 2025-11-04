from typing import List, Dict, Any
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from .config import PipelineConfig
from src.utils.graph_context import build_shortest_path_context

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
        
        # query classification
        query_type = "general_question"
        if getattr(config, "use_query_classification", True) and hasattr(self, "classify_query"):
            query_type = self.classify_query(query)
        if config.verbose:
            print(f"[DEBUG] Query classified as: {query_type}")
        
        # optional query expansion
        if getattr(config, "use_query_expansion", True) and hasattr(self, "_expand_query_with_llm"):
            queries = self._expand_query_with_llm(query, config)
        else: 
            queries = [query]
        if config.verbose:
            print(f"[DEBUG] Expanded into {len(queries)} subqueries: {queries}")

        

        # retrieval stage (implemented/overridden by subclass)
        retrieved, top_nodes = [], []
        for q in queries:
            docs, nodes = self._retrieve(q, query_type, config)
            retrieved.extend(docs)
            top_nodes.extend(nodes or [])
        
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

        top_nodes = self._align_top_nodes_with_docs(retrieved, top_k)

        if getattr(config, "use_shortest_path_context", True) and top_nodes:
            if hasattr(self, "neo4j_uri") and hasattr(self, "neo4j_auth") and hasattr(self, "vectorstore"):
                try:
                    if config.verbose:
                        print("Building shortest-path graph context...")

                    context_docs, path_texts = build_shortest_path_context(
                        query=query,
                        uri=self.neo4j_uri,
                        auth=self.neo4j_auth,
                        vectorstore=self.vectorstore,
                        question_type=query_type,
                        top_nodes=top_nodes,
                        top_k_paths=int(config.top_k_paths) if hasattr(config, "top_k_paths") else 10,
                        top_k_docs_per_path=int(config.top_k_docs_per_path) if hasattr(config, "top_k_docs_per_path") else 10,
                        verbose=config.verbose,
                    )

                    # Add retrieved context to document list
                    if context_docs:
                        if config.verbose:
                            print(f"[DEBUG] Extending with {len(context_docs)} documents")
                        retrieved.extend(context_docs)
                    if path_texts:
                        structural_doc = Document(
                            page_content="\n".join(path_texts),
                            metadata={"type": "graph_structural_context"},
                        )
                        retrieved.append(structural_doc)

                    if config.verbose:
                        print(f"Added {len(context_docs)} docs and {len(path_texts)} structural paths to context.")
                        print(f"Path example: {path_texts[0]}")

                except Exception as e:
                    print(f"[WARN] Shortest-path context building failed: {e}")
            else:
                if config.verbose:
                    print("[INFO] Skipped context building (missing Neo4j or vectorstore).")
        #TODO - Context building (shortest path)
        #   - get shortest paths for top nodes
        #   - get top_k docs per path
        #   - final context: path in text + docs from paths
        # 
        # Maybe mst as well 

        # get combinedNames for eval
        top_function_names = []
        if hasattr(self, "_get_function_names"):
            try:
                top_function_names = self._get_function_names(top_nodes)
            except Exception as e:
                print(f"[WARN] Failed to fetch function names: {e}")

        # generation
        answer = self._generate_answer_from_docs(query, retrieved, config)

        return {
            "answer": answer,
            "top_docs": retrieved,
            "top_functions": top_function_names,
            "top_nodes": top_nodes,
            "query_type": query_type or "default",
        }
    
    def _align_top_nodes_with_docs(self, retrieved: List[Document], top_k: int) -> List[str]:
        """
        Build top_nodes directly from the final ranked documents.
        Ensures node ordering and count match exactly what the model will see.

        Returns:
            List[str]: List of global_ids (e.g., FUNCTION:123) matching retrieved order.
        """
        if not retrieved:
            return []

        top_nodes = []
        for d in retrieved:
            node_id = d.metadata.get("node_id")
            node_type = d.metadata.get("type")
            if not node_id or not node_type:
                continue
            top_nodes.append(self._make_global_id(node_id, node_type))

        return top_nodes[:top_k]

    @staticmethod
    def _make_global_id(node_id: str, node_type: str) -> str:
        """Reconstruct global_id from Qdrant metadata."""
        node_type = node_type.split("_")[0].upper()
        return f"{node_type}:{node_id}"
    
    @abstractmethod
    def _retrieve(self, query: str, config: PipelineConfig):
        """Return (List[Document], query_type: str or None, scored_results: Optional[list])"""

    @abstractmethod
    def _generate_answer_from_docs(self, query: str, docs: List[Document], config: PipelineConfig) -> str:
        """LLM generation"""

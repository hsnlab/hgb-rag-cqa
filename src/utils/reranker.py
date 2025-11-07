from typing import List, Tuple
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document
from neo4j import GraphDatabase

class Reranker:
    def __init__(self, neo4j_config:dict=None, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        # Load cross-encoder for reranking
        self.neo4j_config = neo4j_config
        self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model)
        self.model.to(device)
        self.device = device

    # --------------------
    # Cross-encoder scoring
    # --------------------
    def cross_encoder_score(self, query: str, docs: List[Document]) -> List[float]:
        """Compute cross-encoder relevance scores for (query, doc) pairs."""
        inputs = self.tokenizer(
            [(query, d.page_content) for d in docs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(-1)

        return logits.cpu().tolist()

    # --------------------
    # Graph-aware scoring
    # --------------------
    def graph_score(self, metadata: dict, driver) -> float:
        """
        """
        node_id = metadata.get("node_id", None)
        if node_id:
            query = """
            MATCH (n) WHERE n.global_id = $node_id
            RETURN 
                COUNT {(n)-[r]->()} AS out_degree,
                COUNT {(m)-[r]->(n)} AS in_degree
            """
            with driver.session() as session:
                result = session.run(query, node_id=node_id).single()
                out_degree = result["out_degree"]
                in_degree = result["in_degree"]
                total_degree = out_degree + in_degree

        # Degree penalty (hub nodes are less discriminative)
        degree_score = 1.0 / (1 + np.log1p(total_degree)) if total_degree > 0 else 1.0

        return degree_score
    
    def popularity_score(
        self,
        docs_with_scores: List[Tuple["Document", float]],
        node_id_key: str = "node_id",
        type_key: str = "type",
    ) -> List[Tuple["Document", float]]:
        """
        Rerank documents by multiplying similarity score with node popularity.
        Popularity = (# docs for logical node) / (max # docs for any logical node)

        Logical node is defined by:
            - (issue, node_id)
            - (pr, node_id)
            - (function, node_id)  ← collapsing function_code/docstring/name

        Args:
            docs_with_scores: list of (Document, similarity_score)
            node_id_key: metadata key for node id
            type_key: metadata key for type
        Returns:
            List of (Document, final_score) sorted by final_score desc
        """
        if not docs_with_scores:
            return []

        def canonical_type(t: str) -> str:
            if t and t.startswith("function"):
                return "function"
            elif t and t.startswith("issue"):
                return "issue"
            elif t and t.startswith("pr"):
                return "pr"
            return t

        # Count occurrences by canonical (type, node_id) pairs
        type_node_pairs = [
            (canonical_type(doc.metadata.get(type_key)), doc.metadata.get(node_id_key))
            for doc, _ in docs_with_scores
        ]
        counts = Counter(type_node_pairs)
        max_count = max(counts.values()) if counts else 1

        scored_all = []
        for doc, sim_score in docs_with_scores:
            key = (canonical_type(doc.metadata.get(type_key)), doc.metadata.get(node_id_key))
            popularity = counts[key] / max_count if key[1] and max_count > 0 else 0
            final_score = sim_score * popularity
            scored_all.append((doc, final_score))

        # Sort all scored documents
        scored_all = sorted(scored_all, key=lambda x: x[1], reverse=True)
        return scored_all
    # --------------------
    # Combined reranking
    # --------------------
    def rerank(
        self,
        query: str,
        docs: List[Document],
        alpha: float = 0.7,
        beta: float = 0.3,
        use_graph: bool = False,
        use_popularity: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using a weighted sum of cross-encoder + graph-aware scores,
        with optional popularity reweighting.

        Args:
            query: user query
            docs: list of LangChain Document objects
            alpha: weight for cross-encoder score
            beta: weight for graph-aware score
            use_graph: whether to include graph-aware scoring
            use_popularity: whether to apply popularity-based reweighting
        Returns:
            List of (Document, final_score) sorted by final_score desc
        """
        if not docs:
            return []

        # 1. cross-encoder scores
        cross_scores = self.cross_encoder_score(query, docs)

        # 2. combine with graph scores (if enabled)
        scored_docs = []
        for doc, ce_score in zip(docs, cross_scores):
            if use_graph:
                driver = GraphDatabase.driver(self.neo4j_config["url"], auth=(self.neo4j_config["user"], self.neo4j_config["password"]))
                g_score = self.graph_score(doc.metadata, driver)
                final_score = alpha * ce_score + beta * g_score
            else:
                final_score = ce_score
            scored_docs.append((doc, float(final_score)))

        scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        # 3. apply popularity reweighting (if enabled)
        if use_popularity:
            scored_docs = self.popularity_score(scored_docs)

        return scored_docs
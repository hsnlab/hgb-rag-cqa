from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document

class Reranker:
    def __init__(self, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        # Load cross-encoder for reranking
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
    def graph_score(self, metadata: dict) -> float:
        """
        Compute a score boost/penalty from graph properties.
        Expected metadata keys:
            - 'graph_distance' (int): distance from seed node
            - 'edge_type' (str): type of edge, e.g. CALL, CLUSTER, ISSUE_LINK
            - 'degree' (int): node degree in graph
        """
        distance = metadata.get("graph_distance", 3)
        edge_type = metadata.get("edge_type", "GENERIC")
        degree = metadata.get("degree", 5)

        # Inverse distance weighting (closer = better)
        distance_score = 1.0 / (1 + distance)

        # Edge type weight
        edge_weights = {"CALL": 1.0, "CFG": 0.8, "CLUSTER": 0.6, "ISSUE_LINK": 1.2, "GENERIC": 0.5}
        edge_score = edge_weights.get(edge_type, 0.5)

        # Degree penalty (hub nodes are less discriminative)
        degree_score = 1.0 / (1 + np.log1p(degree))

        return distance_score * edge_score * degree_score

    # --------------------
    # Combined reranking
    # --------------------
    def rerank(
        self,
        query: str,
        docs: List[Document],
        alpha: float = 0.7,
        beta: float = 0.3,
        use_graph: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using a weighted sum of cross-encoder + graph-aware scores.
        Args:
            query: user query
            docs: list of LangChain Document objects
            alpha: weight for cross-encoder score
            beta: weight for graph-aware score
            use_graph: whether to include graph-aware scoring
        Returns:
            List of (Document, final_score) sorted by final_score desc
        """
        if not docs:
            return []

        cross_scores = self.cross_encoder_score(query, docs)

        final_scored_docs = []
        for doc, ce_score in zip(docs, cross_scores):
            if use_graph:
                g_score = self.graph_score(doc.metadata)
                final_score = alpha * ce_score + beta * g_score
            else:
                final_score = ce_score
            final_scored_docs.append((doc, float(final_score)))

        return sorted(final_scored_docs, key=lambda x: x[1], reverse=True)

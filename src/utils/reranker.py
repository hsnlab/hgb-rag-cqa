from typing import List, Tuple
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document
from neo4j import GraphDatabase

class Reranker:
    def __init__(
        self,
        neo4j_config: dict = None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        gnn_projector_path: str = None,
        gnn_encoder_model: str = "intfloat/e5-large-v2",
    ):
        # Load cross-encoder for reranking
        self.neo4j_config = neo4j_config
        self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model)
        self.model.to(device)
        self.device = device

        # Lazily-loaded GNN reranking components
        self.gnn_projector_path = gnn_projector_path
        self.gnn_encoder_model = gnn_encoder_model
        self._projector = None
        self._gnn_encoder = None

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
    # GNN-aware scoring
    # --------------------
    def _ensure_gnn_components(self):
        """Lazy-load the projector and the e5 encoder used to project queries."""
        if self._projector is not None and self._gnn_encoder is not None:
            return
        if not self.gnn_projector_path:
            raise RuntimeError("gnn_projector_path was not provided to Reranker; cannot use GNN scoring.")
        from .query_projector import load_projector
        from langchain_huggingface import HuggingFaceEmbeddings
        self._projector = load_projector(self.gnn_projector_path, device=self.device)
        self._gnn_encoder = HuggingFaceEmbeddings(
            model_name=self.gnn_encoder_model,
            model_kwargs={"device": self.device},
        )

    def _compute_gnn_scores(self, query: str, docs: List[Document]) -> List[float]:
        """Project the query into GNN space and score each doc by cosine similarity.

        Embeddings are fetched from Neo4j in a single batched query for speed.
        Docs whose nodes have no `gnn_embedding` property receive a score of 0.
        """
        self._ensure_gnn_components()

        q_vec = self._gnn_encoder.embed_query(query)
        q_tensor = torch.tensor(q_vec, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_proj = self._projector(q_tensor).squeeze(0).cpu().numpy()

        def _make_gid(doc):
            nid = doc.metadata.get("node_id")
            ntype = doc.metadata.get("type", "")
            if nid is None or not ntype:
                return None
            prefix = ntype.split("_")[0].upper()
            return f"{prefix}:{nid}"

        gids = [_make_gid(d) for d in docs]
        valid_ids = [g for g in gids if g is not None]
        if not valid_ids:
            return [0.0] * len(docs)

        driver = GraphDatabase.driver(
            self.neo4j_config["url"],
            auth=(self.neo4j_config["user"], self.neo4j_config["password"]),
        )
        emb_map = {}
        with driver.session() as session:
            result = session.run(
                "UNWIND $ids AS gid "
                "MATCH (n) WHERE n.global_id = gid AND n.gnn_embedding IS NOT NULL "
                "RETURN gid, n.gnn_embedding AS emb",
                ids=valid_ids,
            )
            for r in result:
                arr = np.array(r["emb"], dtype=np.float32)
                norm = np.linalg.norm(arr)
                emb_map[r["gid"]] = arr / norm if norm > 0 else arr
        driver.close()

        scores = []
        for gid in gids:
            emb = emb_map.get(gid) if gid is not None else None
            if emb is None:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(q_proj, emb)))
        return scores

    # --------------------
    # Combined reranking
    # --------------------
    def rerank(
        self,
        query: str,
        docs: List[Document],
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.0,
        use_graph: bool = False,
        use_gnn: bool = False,
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

        # 2. optional GNN cosine scores (batched once for all docs)
        gnn_scores = None
        if use_gnn:
            try:
                gnn_scores = self._compute_gnn_scores(query, docs)
            except Exception as e:
                print(f"[WARN] GNN scoring failed, falling back without it: {e}")
                gnn_scores = None

        # 3. combine with graph and GNN scores
        scored_docs = []
        for i, (doc, ce_score) in enumerate(zip(docs, cross_scores)):
            final_score = alpha * ce_score
            if use_graph:
                driver = GraphDatabase.driver(self.neo4j_config["url"], auth=(self.neo4j_config["user"], self.neo4j_config["password"]))
                final_score += beta * self.graph_score(doc.metadata, driver)
            if gnn_scores is not None:
                final_score += gamma * gnn_scores[i]
            scored_docs.append((doc, float(final_score)))

        scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        # 4. apply popularity reweighting (if enabled)
        if use_popularity:
            scored_docs = self.popularity_score(scored_docs)

        return scored_docs
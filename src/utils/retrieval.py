# Data
from typing import List, Dict, Any
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from utils.faiss_store import FaissStore
from qdrant_client import models
from transformers import pipeline

class KnowledgeGraphRetriever:
    def __init__(self, vector_store, neo4j_url: str, neo4j_username: str, neo4j_password: str, database: str = "neo4j", query_labels: List[str] = ["general_question", "bug_report", "feature_request", "performance_issue"]):
        """
        Retriever that combines Neo4j graph traversal with FAISS text retrieval.

        Args:
            faiss_store: Your FaissStore instance for text chunk retrieval.
            neo4j_url: Neo4j connection string, e.g. bolt://localhost:7687.
            username: Neo4j username.
            password: Neo4j password.
            database: Neo4j database name (default: "neo4j").
        """
        self.store = vector_store
        self.graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password, database=database,
                                refresh_schema=False)
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"   # Zero-shot or multi-class classification
        )

        self.query_labels = query_labels

    # ------------------
    # Query Classification
    # ------------------
    def classify_query(self, query: str) -> str:
        result = self.classifier(query, self.query_labels, multi_label=False)
        # HuggingFace returns sorted results by score
        return result["labels"][0]

    # ------------------
    # Graph Expansion Helpers
    # ------------------
    def expand_function_neighbors(self, func_ids: List[str], hops: int = 2) -> List[str]:
        """Expand call graph neighborhood using Cypher."""
        query = f"""
        MATCH (f:FUNCTION)
        WHERE f.ID IN $func_ids
        MATCH (f)-[:FUNCTION*1..{hops}]->(nbr:FUNCTION)
        RETURN DISTINCT nbr.ID AS id
        """
        results = self.graph.query(query, params={"func_ids": func_ids})
        return [r["id"] for r in results]

    def expand_cfg_neighbors(self, func_ids: List[str], hops: int = 2) -> List[str]:
        """Expand CFG neighborhood via FUNCTION_SUBGRAPH and SUBGRAPH_FUNCTION edges."""
        query = f"""
        MATCH (f:FUNCTION)
        WHERE f.ID IN $func_ids
        MATCH (f)-[:FUNCTION_SUBGRAPH]->(sg:SUBGRAPH)
        MATCH path = (sg)-[:SUBGRAPH*1..{hops}]->(nbr_sg:SUBGRAPH)
        MATCH (nbr_sg)-[:SUBGRAPH_FUNCTION]->(nbr_func:FUNCTION)
        RETURN DISTINCT nbr_func.ID AS id
        """
        results = self.graph.query(query, params={"func_ids": func_ids})
        return [r["id"] for r in results]

    def functions_linked_to_issues_prs(self, ids: List[str], id_type: str = "issue") -> List[str]:
        """
        Fetch functions linked to given issues or PRs.
        For issues, traverse ISSUE -> PR -> FUNCTION.
        For PRs, traverse PR -> FUNCTION.
        """
        assert id_type in ["issue", "pr"], "id_type must be 'issue' or 'pr'"
        if id_type == "issue":
            query = """
            MATCH (i:ISSUE)-[:ISSUE_PR]->(p:PR)-[:PR_FUNCTION]->(f:FUNCTION)
            WHERE i.ID IN $ids
            RETURN DISTINCT f.ID AS id
            """
        else:  # PR
            query = """
            MATCH (p:PR)-[:PR_FUNCTION]->(f:FUNCTION)
            WHERE p.ID IN $ids
            RETURN DISTINCT f.ID AS id
            """
        results = self.graph.query(query, params={"ids": ids})
        return [r["id"] for r in results]

    # ------------------
    # Retrieval Strategies
    # ------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        top_k = top_k*10  # Over-retrieve to allow for reranking later
        query_type = self.classify_query(query)
        #print(f"Query classified as {query_type}")
        if query_type == "general_question":
            # Step 1: retrieve candidate functions via docstring and function name search
            index_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="function_docstring")
                )
            ])
            func_docs_docstring = self.store.search(query, filter = index_filter, top_k=top_k)
            func_ids_docstring = [d.metadata["node_id"] for d in func_docs_docstring]

            index_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="function_name")
                )
            ])
            func_docs_name = self.store.search(query, filter = index_filter, top_k=top_k)
            func_ids_name = [d.metadata["node_id"] for d in func_docs_name]
            
            func_ids = list(set(func_ids_docstring + func_ids_name))
            func_docs = func_docs_docstring + func_docs_name
            # Step 2: expand neighborhood in KG
            neighbors = self.expand_function_neighbors(func_ids, hops=2)

            # Step 3: fetch neighbor docs using query relevance + filter on func_ids
            neighbor_docs = []
            if neighbors:
                neighbor_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.type",
                            match=models.MatchValue(value="function_code")
                        ),
                        models.FieldCondition(
                            key="metadata.node_id",        
                            match=models.MatchAny(any=neighbors)
                        )
                    ]
                )
                neighbor_docs = self.store.search(
                    query,
                    top_k=len(neighbors),
                    filter=neighbor_filter,
                )

            return func_docs + neighbor_docs, query_type

        elif query_type == "bug_report":
            # Step 1: retrieve issues + PRs
            issue_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="issue_body")
                )
            ])
            issue_docs = self.store.search(query, filter = issue_filter, top_k=top_k)
            pr_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="pr_body")
                )
            ])
            pr_docs = self.store.search(query, filter = pr_filter, top_k=top_k)

            # Step 2: expand to functions linked to these issues/PRs
            issue_ids = [d.metadata["node_id"] for d in issue_docs]
            pr_ids = [d.metadata["node_id"] for d in pr_docs]

            func_ids = self.functions_linked_to_issues_prs(issue_ids, id_type="issue")
            func_ids += self.functions_linked_to_issues_prs(pr_ids, id_type="pr")

            # Step 3: expand call graph neighborhood
            neighbors = self.expand_function_neighbors(func_ids, hops=2)

            # Step 4: fetch function docs with filter on func_ids + neighbors
            func_docs = []
            target_ids = func_ids + neighbors
            if target_ids:
                filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.type",
                            match=models.MatchValue(value="function_code")
                        ),
                        models.FieldCondition(
                            key="metadata.node_id",        
                            match=models.MatchAny(any=target_ids)
                        )
                    ]
                )
                func_docs = self.store.search(
                    query,
                    top_k=len(target_ids),
                    filter=filter,
                )

            return issue_docs + pr_docs + func_docs, query_type

        elif query_type == "feature_request":
            # Cluster-level search
            index_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="semantic_cluster")
                )
            ])
            cluster_docs = self.store.search(query, filter=index_filter, top_k=3)
            cluster_ids = [d.metadata["node_id"] for d in cluster_docs]

            func_ids = self.graph.query("""
                MATCH (c:CLUSTER)-[:CLUSTER_FUNCTION]->(f:FUNCTION)
                WHERE c.ID IN $cluster_ids
                RETURN f.ID AS id
            """, params={"cluster_ids": cluster_ids})
            
            func_ids = [r["id"] for r in func_ids]

            filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.node_id",
                    match=models.MatchAny(any=func_ids)
                )
            ]) if func_ids else None
            return self.store.search(query, top_k=top_k, filter=filter), query_type

        elif query_type == "performance_issue":
            # Cluster search + performance-tagged issues/PRs
            index_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="semantic_cluster")
                )
            ])
            cluster_docs = self.store.search(query, filter=index_filter, top_k=5)
            cluster_ids = [d.metadata["node_id"] for d in cluster_docs]

            func_ids = self.graph.query("""
                MATCH (c:CLUSTER)-[:CLUSTER_FUNCTION]->(f:FUNCTION)
                WHERE c.ID IN $cluster_ids
                RETURN f.ID AS id
            """, params={"cluster_ids": cluster_ids})

            func_ids = [r["id"] for r in func_ids]
            
            # Expand via CFG edges
            cfg_neighbors = self.expand_cfg_neighbors(func_ids, hops=2)
            all_func_ids = func_ids + cfg_neighbors
            
            filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.node_id",
                    match=models.MatchAny(any=all_func_ids)
                ),
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="function_code")
                ),

            ]) if func_ids else None
            return self.store.search(query, top_k=top_k, filter=filter), query_type

        else:
            return [], "Error"

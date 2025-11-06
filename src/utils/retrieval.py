# Data
from typing import List, Dict, Any
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from qdrant_client import models
from transformers import pipeline

class KnowledgeGraphRetriever:
    def __init__(self, vector_store, neo4j_url: str, neo4j_username: str, neo4j_password: str, database: str = "neo4j"):
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
        
    # ------------------
    # Graph Expansion Helpers
    # ------------------
    def expand_function_neighbors(self, func_ids: List[str], hops: int = 2) -> List[str]:
        """Expand call graph neighborhood using Cypher."""
        query = f"""
        MATCH (f:FUNCTION)
        WHERE f.global_id IN $func_ids
        MATCH (f)-[:FUNCTION*1..{hops}]->(nbr:FUNCTION)
        RETURN DISTINCT nbr.global_id AS id
        """
        results = self.graph.query(query, params={"func_ids": func_ids})
        return [r["id"] for r in results]

    def expand_cfg_neighbors(self, func_ids: List[str], hops: int = 2) -> List[str]:
        """Expand CFG neighborhood via FUNCTION_SUBGRAPH and SUBGRAPH_FUNCTION edges."""
        query = f"""
        MATCH (f:FUNCTION)
        WHERE f.global_id IN $func_ids
        MATCH (f)-[:FUNCTION_SUBGRAPH]->(sg:SUBGRAPH)
        MATCH path = (sg)-[:SUBGRAPH*1..{hops}]->(nbr_sg:SUBGRAPH)
        MATCH (nbr_sg)-[:SUBGRAPH_FUNCTION]->(nbr_func:FUNCTION)
        RETURN DISTINCT nbr_func.global_id AS id
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
            WHERE i.global_id IN $ids
            RETURN DISTINCT f.global_id AS id
            """
        else:  # PR
            query = """
            MATCH (p:PR)-[:PR_FUNCTION]->(f:FUNCTION)
            WHERE p.global_id IN $ids
            RETURN DISTINCT f.global_id AS id
            """
        results = self.graph.query(query, params={"ids": ids})
        return [r["id"] for r in results]

    # ------------------
    # Retrieval Strategies
    # ------------------
    def retrieve(self, query: str, top_k: int = 5, query_type:str = "general_question") -> List[Document]:
        if query_type == "general_question":
            # Step 1: retrieve candidate functions via docstring and function name search
            index_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchAny(any=["function_docstring", "function_code"])
                )
            ])
            func_docs_code = self.store.search(query, filter = index_filter, top_k=top_k)

            index_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchAny(any=["function_name", "function_split_names"])
                )
            ])
            func_docs_name = self.store.search(query, filter = index_filter, top_k=top_k)
            
            func_docs = func_docs_code + func_docs_name
            func_global_ids = [d.metadata["node_id"] for d in func_docs]
            # Step 2: expand neighborhood in KG
            neighbors = self.expand_function_neighbors(func_global_ids, hops=2)
            top_node_ids = list(set(func_global_ids + neighbors))

            # Step 3: fetch neighbor docs using query relevance + filter on func_ids
            neighbor_docs = []
            if neighbors:
                neighbor_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.type",
                            match=models.MatchAny(any=["function_docstring", "function_code"])
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

            return func_docs + neighbor_docs, top_node_ids

        elif query_type == "bug_report":
            # Step 1: retrieve issues + PRs
            issue_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchAny(any=["issue_body","issue_title"])
                )
            ])
            issue_docs = self.store.search(query, filter = issue_filter, top_k=top_k)
            pr_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchAny(any=["pr_body", "pr_title"])
                )
            ])
            pr_docs = self.store.search(query, filter = pr_filter, top_k=top_k)

            # Step 2: expand to functions linked to these issues/PRs
            issue_global_ids = [d.metadata["node_id"] for d in issue_docs]
            pr_global_ids = [d.metadata["node_id"] for d in pr_docs]

            func_ids = self.functions_linked_to_issues_prs(issue_global_ids, id_type="issue")
            func_ids += self.functions_linked_to_issues_prs(pr_global_ids, id_type="pr")

            all_docs = issue_docs + pr_docs

            # Step 3: expand call graph neighborhood
            neighbors = self.expand_function_neighbors(func_ids, hops=2)
            top_node_ids = list(set(issue_global_ids + pr_global_ids + func_ids + neighbors))

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
                all_docs += func_docs

            return all_docs, top_node_ids

        elif query_type == "feature_request":
            # Step 1: cluster-level search
            cluster_filter = models.Filter(must=[
                models.FieldCondition(key="metadata.type", match=models.MatchValue(value="semantic_cluster"))
            ])
            cluster_docs = self.store.search(query, filter=cluster_filter, top_k=3)
            cluster_global_ids = [d.metadata["node_id"] for d in cluster_docs]

            # Step 2: fetch linked functions
            func_results = self.graph.query("""
                MATCH (c:CLUSTER)-[:CLUSTER_FUNCTION]->(f:FUNCTION)
                WHERE c.global_id IN $cluster_ids
                RETURN f.global_id AS id
            """, params={"cluster_ids": cluster_global_ids})
            func_ids = [r["id"] for r in func_results]
            top_node_ids = list(set(cluster_global_ids + func_ids))

            # Step 3: retrieve function docs
            func_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.node_id",
                    match=models.MatchAny(any=func_ids)
                ),
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchAny(any=["function_code", "function_docstring"])
                )
            ]) if func_ids else None
            func_docs = self.store.search(query, top_k=top_k, filter=func_filter)
            return func_docs, top_node_ids

        elif query_type == "performance_issue":
            # Step 1: retrieve clusters
            cluster_filter = models.Filter(must=[
                models.FieldCondition(key="metadata.type", match=models.MatchValue(value="semantic_cluster"))
            ])
            cluster_docs = self.store.search(query, filter=cluster_filter, top_k=5)
            cluster_global_ids = [d.metadata["node_id"] for d in cluster_docs]

            # Step 2: get functions in clusters
            func_results = self.graph.query("""
                MATCH (c:CLUSTER)-[:CLUSTER_FUNCTION]->(f:FUNCTION)
                WHERE c.global_id IN $cluster_ids
                RETURN f.global_id AS id
            """, params={"cluster_ids": cluster_global_ids})
            func_ids = [r["id"] for r in func_results]

            # Step 3: expand CFG neighborhood
            cfg_neighbors = self.expand_cfg_neighbors(func_ids, hops=2)
            top_node_ids = list(set(cluster_global_ids + func_ids + cfg_neighbors))

            func_filter = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.node_id",
                    match=models.MatchAny(any= func_ids + cfg_neighbors)
                ),
                models.FieldCondition(
                    key="metadata.type",
                    match=models.MatchValue(value="function_code")
                ),
            ]) if func_ids else None
            func_docs = self.store.search(query, top_k=top_k, filter=func_filter)
            return func_docs, top_node_ids

        else:
            return [], "Error", []

    @staticmethod
    def _make_global_id(node_id: str, node_type: str) -> str:
        """
        Normalize node_type like 'function_code' → 'FUNCTION' and compose global_id.
        """        
        node_type = node_type.split("_")[0].upper()
        return f"{node_type}:{node_id}"

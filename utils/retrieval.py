# Data
from typing import List, Dict, Any
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from faiss_store import FaissStore



class Retrieval():

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.store = FaissStore(model_name=model_name)

    def index_data(self, issues_df, prs_df, code_df):
        self.store.add_issues(issues_df)
        self.store.add_prs(prs_df)
        self.store.add_code(code_df)

    def retrieve(self, query, source="issue", top_k=5):
        results = self.store.search(query, index_type=source, top_k=top_k)
        return [
            {"text": doc.page_content, **doc.metadata}
            for doc in results
        ]

class KnowledgeGraphRetriever:
    def __init__(self, faiss_store, neo4j_url: str, username: str, password: str, database: str = "neo4j"):
        """
        Retriever that combines Neo4j graph traversal with FAISS text retrieval.

        Args:
            faiss_store: Your FaissStore instance for text chunk retrieval.
            neo4j_url: Neo4j connection string, e.g. bolt://localhost:7687.
            username: Neo4j username.
            password: Neo4j password.
            database: Neo4j database name (default: "neo4j").
        """
        self.store = faiss_store
        self.graph = Neo4jGraph(url=neo4j_url, username=username, password=password, database=database)

    # ------------------
    # Query Classification
    # ------------------
    def classify_query(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["bug", "error", "fix", "fail"]):
            return "bug_report"
        elif any(word in query_lower for word in ["feature", "add", "support"]):
            return "feature_request"
        elif any(word in query_lower for word in ["slow", "performance", "optimize", "latency"]):
            return "performance_issue"
        else:
            return "general_qa"

    # ------------------
    # Graph Expansion Helpers
    # ------------------
    def expand_function_neighbors(self, func_ids: List[str], hops: int = 2) -> List[str]:
        """Expand call graph neighborhood using Cypher."""
        query = f"""
        MATCH (f:Function)
        WHERE f.func_id IN $func_ids
        MATCH (f)-[:CALL*1..{hops}]->(nbr:Function)
        RETURN DISTINCT nbr.func_id AS id
        """
        results = self.graph.query(query, params={"func_ids": func_ids})
        return [r["id"] for r in results]

    def functions_linked_to_issues_prs(self, ids: List[str], id_type: str = "issue") -> List[str]:
        """Fetch functions linked to given issues/PRs."""
        label = "Issue" if id_type == "issue" else "PR"
        field = "issue_number" if id_type == "issue" else "pr_number"
        query = f"""
        MATCH (n:{label})-[:RELATED_TO]->(f:Function)
        WHERE n.{field} IN $ids
        RETURN DISTINCT f.func_id AS id
        """
        results = self.graph.query(query, params={"ids": ids})
        return [r["id"] for r in results]

    # ------------------
    # Retrieval Strategies
    # ------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_type = self.classify_query(query)

        if query_type == "general_qa":
            # Step 1: retrieve candidate functions via FAISS
            func_docs = self.store.search(query, index_type="code", top_k=top_k)
            func_ids = [d.metadata["func_id"] for d in func_docs]

            # Step 2: expand neighborhood in KG
            neighbors = self.expand_function_neighbors(func_ids, hops=2)

            # Step 3: fetch neighbor docs using query relevance + filter on func_ids
            neighbor_docs = []
            if neighbors:
                neighbor_docs = self.store.search(
                    query,
                    index_type="code",
                    top_k=len(neighbors),
                    filter={"func_id": {"$in": neighbors}},
                )

            return func_docs + neighbor_docs

        elif query_type == "bug_report":
            # Step 1: retrieve issues + PRs
            issue_docs = self.store.search(query, index_type="issue", top_k=top_k)
            pr_docs = self.store.search(query, index_type="pr", top_k=top_k)

            # Step 2: expand to functions linked to these issues/PRs
            issue_ids = [d.metadata["issue_number"] for d in issue_docs]
            pr_ids = [d.metadata["pr_number"] for d in pr_docs]

            func_ids = self.functions_linked_to_issues_prs(issue_ids, id_type="issue")
            func_ids += self.functions_linked_to_issues_prs(pr_ids, id_type="pr")

            # Step 3: expand call graph neighborhood
            neighbors = self.expand_function_neighbors(func_ids, hops=2)

            # Step 4: fetch function docs with filter on func_ids + neighbors
            func_docs = []
            target_ids = func_ids + neighbors
            if target_ids:
                func_docs = self.store.search(
                    query,
                    index_type="code",
                    top_k=len(target_ids),
                    filter={"func_id": {"$in": target_ids}},
                )

            return issue_docs + pr_docs + func_docs

        elif query_type == "feature_request":
            # TODO: implement retrieval strategy for feature requests
            return []

        elif query_type == "performance_issue":
            # TODO: implement retrieval strategy for performance issues
            return []

        else:
            return []

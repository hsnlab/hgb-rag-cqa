# neo4j_mcp_server.py
from fastmcp import FastMCP
from neo4j import GraphDatabase
from typing import List

# Create MCP server
mcp = FastMCP("neo4j-retriever")

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

@mcp.tool()
def expand_function_neighbors(func_ids: List[str], hops: int = 2) -> List[str]:
    """
    Expand function neighborhood in the call graph.
    """
    query = f"""
    MATCH (f:FUNCTION) WHERE f.ID IN $func_ids
    MATCH (f)-[:FUNCTION*1..{hops}]->(nbr:FUNCTION)
    RETURN DISTINCT nbr.ID AS id
    """
    with driver.session() as session:
        results = session.run(query, func_ids=func_ids)
        return [r["id"] for r in results]

@mcp.tool()
def expand_cfg_neighbors(func_ids: List[str], hops: int = 2) -> List[str]:
    """
    Expand CFG neighborhood via FUNCTION_SUBGRAPH and SUBGRAPH_FUNCTION.
    """
    query = f"""
    MATCH (f:FUNCTION) WHERE f.ID IN $func_ids
    MATCH (f)-[:FUNCTION_SUBGRAPH]->(sg:SUBGRAPH)
    MATCH path = (sg)-[:SUBGRAPH*1..{hops}]->(nbr_sg:SUBGRAPH)
    MATCH (nbr_sg)-[:SUBGRAPH_FUNCTION]->(nbr_func:FUNCTION)
    RETURN DISTINCT nbr_func.ID AS id
    """
    with driver.session() as session:
        results = session.run(query, func_ids=func_ids)
        return [r["id"] for r in results]

@mcp.tool()
def functions_linked_to_issues_prs(ids: List[str], id_type: str = "issue") -> List[str]:
    """
    Fetch functions linked to given issues or PRs.
    """
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
    with driver.session() as session:
        results = session.run(query, ids=ids)
        return [r["id"] for r in results]

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8765, path="/mcp")
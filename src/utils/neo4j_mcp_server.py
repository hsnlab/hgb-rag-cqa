# neo4j_mcp_server.py
from fastmcp import FastMCP
from neo4j import GraphDatabase
from typing import List, Dict, Any
import json

# Create MCP server
mcp = FastMCP("neo4j-retriever")

neo4j_config_path = "_/neo4j_config.json"
try:
    with open(neo4j_config_path, "r") as f:
        neo4j_config = json.load(f)
except FileNotFoundError:
    print(f"Error: Neo4j config file not found at {neo4j_config_path}")
except json.JSONDecodeError:
    print(f"Error: Neo4j config file is not valid JSON: {neo4j_config_path}")

driver = GraphDatabase.driver(neo4j_config["url"], auth=(neo4j_config["user"], neo4j_config["password"]))

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
    
@mcp.tool()
def get_node_info(node_id:str, field:str) -> str:
    """
    """
    node_type = node_id.split(":")[0]
    node_type = node_type.upper()
    query = f"""
    MATCH (n:{node_type})
    WHERE n.ID = $node_id
    RETURN n.{field} as field
    """
    with driver.session() as session:
        try:
            result = session.run(query,node_id=node_id)
        except:
            print(f"[DEBUG] Failed to run get_node_info endpoint for node: {node_id}, field: {field}")
            result = ""
    return result

@mcp.tool()
def minimal_spanning_tree(node_ids: List[str]) -> List[Dict[str, str]]:
    """
    Compute the minimal spanning tree (MST) connecting the given nodes in the Neo4j knowledge graph.
    - Works for arbitrary node labels.
    - Considers all relationships among these nodes.
    - Treats all edges as weight=1 (unweighted MST).
    Returns: a list of {"source": <id>, "target": <id>} edges in the MST.
    """
    with driver.session() as session:
        # -- Cleanup from previous runs (if it exists)
        session.run("CALL gds.graph.drop('mst_subgraph', false)")  

        # -- 1. Project subgraph into GDS (label-agnostic, relationship-agnostic)
        projection_query = """
        CALL gds.graph.project.cypher(
            'mst_subgraph',
            'MATCH (n) WHERE n.ID IN $node_ids RETURN id(n) AS id',
            'MATCH (a)-[r]-(b)
             WHERE a.ID IN $node_ids AND b.ID IN $node_ids
             RETURN id(a) AS source, id(b) AS target',
            { parameters: { node_ids: $node_ids } }
        )
        YIELD graphName, nodeCount, relationshipCount
        """
        session.run(projection_query, node_ids=node_ids)

        # -- 2. Compute MST (treating all edges as equal-weight)
        mst_query = """
        CALL gds.spanningTree.minimum.stream('mst_subgraph')
        YIELD sourceNodeId, targetNodeId
        RETURN gds.util.asNode(sourceNodeId).ID AS source,
               gds.util.asNode(targetNodeId).ID AS target
        """
        results = session.run(mst_query)
        edges = [{"source": r["source"], "target": r["target"]} for r in results]

        # -- 3. Clean up projection to free memory
        session.run("CALL gds.graph.drop('mst_subgraph', false)")

    return edges

@mcp.tool()
def get_edge_context(edges: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Given a list of edges [{source, target}], fetch structured information
    about the connected nodes and their relationship types.
    This is used to enrich a minimal spanning tree result safely.
    """
    with driver.session() as session:
        try:
            query = """
            UNWIND $edges AS e
            MATCH (a {ID: e.source})-[r]-(b {ID: e.target})
            RETURN 
                a.ID AS source_id, labels(a) AS source_labels,
                b.ID AS target_id, labels(b) AS target_labels,
                type(r) AS rel_type
            """
            results = session.run(query, edges=edges)
            enriched = []
            for record in results:
                enriched.append({
                    "source_id": record["source_id"],
                    "source_labels": record["source_labels"],
                    "target_id": record["target_id"],
                    "target_labels": record["target_labels"],
                    "rel_type": record["rel_type"],
                })
            return enriched
        except Exception as e:
            print(f"[ERROR] get_edge_context failed: {e}")
            return []

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8765, path="/mcp")
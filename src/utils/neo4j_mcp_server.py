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
    WHERE n.global_id = $node_id
    RETURN n.{field} as field
    """
    with driver.session() as session:
        try:
            result = session.run(query,node_id=node_id).single()
            print(f"[DEBUG] Got results for node: {node_id}: {result}")
            if result:
                field_value = result['field']
                if field_value is not None:
                    return str(field_value)
                else:
                    print(f"[DEBUG] Got null value for node: {node_id}, field: {field}")
                    return ""
            else:
                return ""
        except Exception as e:
            print(f"[DEBUG] Failed to run get_node_info endpoint for node: {node_id}, field: {field}:{e}")
            return ""

@mcp.tool()
def minimal_spanning_tree(node_ids: List[str]) -> List[Dict[str, str]]:
    """
    Compute the minimal spanning tree (MST) connecting the given nodes in the Neo4j knowledge graph.
    - Works for arbitrary node labels (using global_id).
    - Considers all relationships among these nodes.
    - Treats all edges as weight=1 (unweighted MST).
    Returns: a list of {"source": <global_id>, "target": <global_id>} edges in the MST.
    """
    
    if not node_ids:
        return []

    with driver.session() as session:
        graph_name = 'mst_subgraph' # Define graph name once
        
        try:
            # Drop graph if it exists from a previous run
            session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception as e:
            print(f"[DEBUG] gds.graph.drop failed (likely harmless): {e}")

        # --- FIX 1: Revert to gds.graph.project.cypher ---
        # And keep using elementId() to avoid the other deprecation warning
        projection_query = f"""
        CALL gds.graph.project.cypher(
            '{graph_name}',
            
            'MATCH (n) WHERE n.global_id IN $node_ids RETURN id(n) AS id',
            
            'MATCH (a)-[r]-(b)
             WHERE a.global_id IN $node_ids AND b.global_id IN $node_ids
             RETURN id(a) AS source, id(b) AS target, 1 as weight',
             
            {{ parameters: {{ node_ids: $node_ids }} }}
        )
        YIELD graphName, nodeCount, relationshipCount
        """
        
        try:
            # Pass node_ids as a parameter to the session.run
            session.run(projection_query, node_ids=node_ids)
        except Exception as e:
            print(f"[ERROR] GDS projection failed: {e}")
            # Ensure cleanup even if projection fails
            try:
                session.run(f"CALL gds.graph.drop('{graph_name}', false)")
            except Exception:
                pass # Ignore cleanup error
            return [] 

        # --- FIX 2: Get start node ELEMENT ID using elementId() ---
        start_node_global_id = node_ids[0]
        # Use elementId()
        start_id_query = "MATCH (n) WHERE n.global_id = $global_id RETURN elementId(n) AS id"
        start_id_result = session.run(start_id_query, global_id=start_node_global_id).single()
        
        if not start_id_result:
            print(f"[ERROR] MST start node {start_node_global_id} not found in DB.")
            session.run(f"CALL gds.graph.drop('{graph_name}', false)") # Cleanup
            return []
            
        start_node_element_id = start_id_result['id'] 

        # --- FIX 3: Use 'sourceNode' in the MST query ---
        mst_query = f"""
        CALL gds.spanningTree.stream('{graph_name}', {{
            sourceNode: $start_node_element_id,
            relationshipWeightProperty: 'weight'
        }})
        YIELD nodeId, parentId, weight
        WHERE parentId IS NOT NULL 
        RETURN gds.util.asNode(nodeId).global_id AS source,
               gds.util.asNode(parentId).global_id AS target
        """
        
        edges = []
        try:
            # Pass the element ID string
            results = session.run(mst_query, start_node_element_id=start_node_element_id)
            edges = [{"source": r["source"], "target": r["target"]} for r in results]
        except Exception as e:
            print(f"[ERROR] GDS spanningTree failed: {e}")
            # Fall through to cleanup

        finally:
            # -- Always clean up the projected graph --
            try:
                session.run(f"CALL gds.graph.drop('{graph_name}', false)")
            except Exception as e:
                 print(f"[WARN] Failed to drop GDS graph '{graph_name}': {e}")


    return edges

@mcp.tool()
def get_edge_context(edges: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Given a list of edges [{source, target}], fetch structured information
    about the connected nodes and their relationship types.
    This is used to enrich a minimal spanning tree result safely.
    """
    if not edges:
        return []

    with driver.session() as session:
        try:
            query = """
            UNWIND $edges AS e
            MATCH (a {global_id: e.source})-[r]-(b {global_id: e.target})
            RETURN 
                a.global_id AS source_id, labels(a) AS source_labels,
                b.global_id AS target_id, labels(b) AS target_labels,
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
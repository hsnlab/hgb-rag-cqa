from pathlib import Path
from fastmcp import FastMCP
import json
from typing import Optional, Dict, List
from src.utils.qdrant_store import QdrantStore

# Create MCP server
mcp = FastMCP("qdrant-store")
qdrant_config_path = "_/qdrant_config.json"
try:
    with open(qdrant_config_path, "r") as f:
        qdrant_config = json.load(f)
except FileNotFoundError:
    print(f"Error: Qdrant config file not found at {qdrant_config_path}")
except json.JSONDecodeError:
    print(f"Error: Qdrant config file is not valid JSON: {qdrant_config_path}")

neo4j_config_path = "_/neo4j_config.json"
try:
    with open(neo4j_config_path, "r") as f:
        neo4j_config = json.load(f)
except FileNotFoundError:
    print(f"Error: Neo4j config file not found at {neo4j_config_path}")
except json.JSONDecodeError:
    print(f"Error: Neo4j config file is not valid JSON: {neo4j_config_path}")


print(f"Loaded configs: {qdrant_config}\n{neo4j_config}")
# Initialize your QdrantStore (read-only)

qdrant_store = QdrantStore(
            model_name=qdrant_config["model_name"],
            qdrant_url=qdrant_config["url"],
            collection_name=qdrant_config["collection"],
            api_key=qdrant_config["api_key"],
            distance_type=qdrant_config["distance"]
        )

@mcp.tool()
def qdrant_search(query: str, top_k: int = 5, metadata_filter: Optional[Dict] = None) -> List[Dict]:
    """
    Search Qdrant for documents matching a query. Read-only tool.
    """
    print(f"[DEBUG] Searching for: '{query}' with top_k={top_k} and filter={metadata_filter}")

    try:
        results = qdrant_store.search_with_scores(query, top_k=top_k, filter=metadata_filter)
        print(f"[DEBUG] Got {len(results)} results")
        
        for i, (doc, score) in enumerate(results):
            print(f"[DEBUG] Result {i}: score={score}, content_length={len(doc.page_content)}")
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    except Exception as e:
        print(f"[DEBUG] Error during search: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8766,path="/mcp")
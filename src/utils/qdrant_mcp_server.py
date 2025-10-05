from pathlib import Path
from fastmcp import FastMCP
from typing import Optional, Dict, List
from qdrant_store import QdrantStore  # <-- import your wrapper class

# Create MCP server
mcp = FastMCP("qdrant-store")
qdrant_key_path = Path(__file__).resolve().parents[2] / "_" / "drant_api_key.txt"
with open(qdrant_key_path, "r") as f:
    qdrant_apikey = f.read().strip()

# Initialize your QdrantStore (read-only)
qdrant_store = QdrantStore(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    qdrant_url="http://localhost:6333",
    collection_name="rag_collection",
    api_key=qdrant_apikey,
    neo4j_uri="bolt://localhost:7687",
    neo4j_auth=("neo4j", "password")
)

@mcp.tool()
def qdrant_search(query: str, top_k: int = 5, metadata_filter: Optional[Dict] = None) -> List[Dict]:
    """
    Search Qdrant for documents matching a query. Read-only tool.
    """
    results = qdrant_store.search_with_scores(query, top_k=top_k, filter=metadata_filter)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        }
        for doc, score in results
    ]

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8766,path="/mcp")
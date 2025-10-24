from pathlib import Path
from fastmcp import FastMCP
from typing import Optional, Dict, List
from src.utils.qdrant_store import QdrantStore  # <-- import your wrapper class

# Create MCP server
mcp = FastMCP("qdrant-store")
qdrant_key_path = Path(__file__).resolve().parents[2] / "_" / "drant_api_key.txt"
with open(qdrant_key_path, "r") as f:
    qdrant_apikey = f.read().strip()

# Initialize your QdrantStore (read-only)
qdrant_store = QdrantStore(
    model_name="microsoft/codebert-base",
    qdrant_url="http://localhost:6333",
    collection_name="rag_collection_codebert-base_cosine",
    api_key=qdrant_apikey,
    neo4j_uri="bolt://localhost:7687",
    neo4j_auth=("neo4j", "password")
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
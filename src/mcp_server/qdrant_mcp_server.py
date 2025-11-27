import asyncio
import json
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from src.utils.qdrant_store import QdrantStore

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_qdrant_server")

with open("./_/drant_api_key.txt", "r") as f:
    qdrant_api_key = f.read().strip()

# Inicializáljuk a QdrantStore objektumot (ez belül létrehozza a QdrantClientet)
store = QdrantStore(
    model_name="BAAI/bge-small-en-v1.5",
    qdrant_url="http://localhost:6333",
    collection_name="rag_collection_bge_small",
    api_key=qdrant_api_key,
    distance_type="cosine",
)

# MCP + FastAPI
app = FastAPI()

registered_tools = {}

# -------------------------
# MCP TOOL DECORATOR
# -------------------------
def register_tool(func):
    registered_tools[func.__name__] = func
    return func

# -------------------------
# TOOL DEFINITIONS
# -------------------------
@register_tool
def qdrant_search(query: str, top_k: int = 5):
    """Search documents in Qdrant via QdrantStore"""
    results = store.search_with_scores(query, top_k=top_k)
    return [{"text": d.page_content, "score": s, "metadata": d.metadata} for d, s in results]


@register_tool
def qdrant_clear():
    """Clear Qdrant collection"""
    store.clear_collection()
    return {"status": "cleared"}


@register_tool
def qdrant_info():
    """Get collection info"""
    info = store.get_collection_info()
    return {"status": "ok", "info": str(info)}


# -------------------------
# MCP SSE endpoint
# -------------------------
@app.get("/mcp")
async def event_stream():
    async def event_generator():
        # --- Heartbeat ---
        while True:
            yield ": keep-alive\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# -------------------------
# MCP RPC endpoint (POST /mcp)
# -------------------------
@app.post("/mcp")
async def mcp_call(request: Request):
    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})
    id_ = body.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": id_,
            "result": {"protocolVersion": "1.0", "serverInfo": {"name": "Qdrant-MCP-Server", "version": "1.0.0"}, "capabilities": {"tools":{}}}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": id_,
            "result": {"tools": [{"name": n, "description": f.__doc__ or ""} for n, f in registered_tools.items()]},
        }

    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        func = registered_tools.get(tool_name)
        if not func:
            return {"jsonrpc": "2.0", "id": id_, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}

        try:
            result = func(**args)
            return {"jsonrpc": "2.0", "id": id_, "result": result}
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {e}")
            return {"jsonrpc": "2.0", "id": id_, "error": {"code": -32000, "message": str(e)}}

    return {"jsonrpc": "2.0", "id": id_, "error": {"code": -32601, "message": f"Unknown method {method}"}}


# -------------------------
# START
# -------------------------
if __name__ == "__main__":
    print("🚀 Starting MCP Qdrant Server on http://0.0.0.0:8001/mcp")
    uvicorn.run(app, host="0.0.0.0", port=8001)

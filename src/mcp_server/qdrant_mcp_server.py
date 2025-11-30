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

# Inicializáljuk a QdrantStore objektumot (ez belül létrehozza a QdrantClientet)
store = QdrantStore(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    qdrant_url="http://localhost:6333",
    collection_name="rag_collection_codebert-base_cosine"
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
        # --- READY event (kell a CrewAI-nek) ---
        yield "event: ready\ndata: {\"status\": \"ok\"}\n\n"

        # --- MCP handshake ---
        init_msg = {
            "jsonrpc": "2.0",
            "method": "session/initialized",
            "params": {"protocol": "MCP/1.0"},
        }
        yield f"data: {json.dumps(init_msg)}\n\n"

        # --- Tool lista ---
        tools_data = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {
                "tools": [
                    {"name": name, "description": func.__doc__ or ""}
                    for name, func in registered_tools.items()
                ]
            },
        }
        yield f"data: {json.dumps(tools_data)}\n\n"

        # --- Heartbeat ---
        while True:
            heartbeat = {
                "jsonrpc": "2.0",
                "method": "notifications/heartbeat",
                "params": {"status": "alive"},
            }
            yield f"data: {json.dumps(heartbeat)}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# -------------------------
# MCP RPC endpoint (POST /mcp)
# -------------------------
@app.post("/mcp")
async def mcp_call(request: Request):
    body = await request.json()
    method = body.get("method")
    id_ = body.get("id")

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

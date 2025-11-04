import requests
import json
import time

url = "http://localhost:8766/mcp"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream"
}

session = requests.Session()

# Step 1: Initialize
print("=== Step 1: Initialize ===")
init_payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    }
}

response = session.post(url, headers=headers, json=init_payload)
session_id = response.headers.get("mcp-session-id")
print(f"Session ID: {session_id}")

# Parse initialization response
for line in response.text.split('\n'):
    if line.startswith('data: '):
        data = json.loads(line[6:])
        print("Init response:", json.dumps(data, indent=2))

# Step 2: Send initialized notification
print("\n=== Step 2: Send initialized notification ===")
headers["mcp-session-id"] = session_id

initialized_payload = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
}

response = session.post(url, headers=headers, json=initialized_payload)
print(f"Initialized notification status: {response.status_code}")

# Give it a moment to process
time.sleep(0.5)

# Step 3: List available tools (optional but good to verify)
print("\n=== Step 3: List tools ===")
list_tools_payload = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
}

response = session.post(url, headers=headers, json=list_tools_payload)
for line in response.text.split('\n'):
    if line.startswith('data: '):
        data = json.loads(line[6:])
        print("Available tools:", json.dumps(data, indent=2))

# Step 4: Call the tool
print("\n=== Step 4: Call qdrant_search tool ===")
tool_payload = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "qdrant_search",
        "arguments": {
            "query": "pca.fit",
            "top_k": 3
        }
    }
}

response = session.post(url, headers=headers, json=tool_payload)
print(f"Tool call status: {response.status_code}")
print("\nTool call response:")
for line in response.text.split('\n'):
    if line.startswith('data: '):
        data = json.loads(line[6:])
        print(json.dumps(data, indent=2))
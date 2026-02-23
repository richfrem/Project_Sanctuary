import httpx
import json
import os
import sys
import time

# Standardized Environment Variables
GATEWAY_URL = os.getenv("MCP_GATEWAY_URL", "https://localhost:4444")
# Use the standardized token variable
API_TOKEN = os.getenv("MCPGATEWAY_BEARER_TOKEN")

if not API_TOKEN:
    print("❌ ERROR: MCPGATEWAY_BEARER_TOKEN not found in environment.")
    sys.exit(1)

# Security: Disable SSL verification for local dev if needed
verify_ssl = os.getenv("MCP_GATEWAY_VERIFY_SSL", "false").lower() == "true"

def run_rpc(method, params, id=1, timeout=60.0):
    """Call the Gateway's JSON-RPC endpoint."""
    url = f"{GATEWAY_URL}/rpc"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": id
    }
    
    with httpx.Client(verify=verify_ssl) as client:
        try:
            resp = client.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                print(f"❌ HTTP Error {resp.status_code}: {resp.text}")
                return None
            return resp.json()
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return None

def test_learning_loop():
    print("--- PROTOCOL 125 DEEP VERIFICATION LOOP ---")
    print(f"Target Gateway: {GATEWAY_URL}")
    print(f"Verify SSL: {verify_ssl}")
    
    # 0. Tool Discovery
    print("\n0. Discovering Tools via /rpc...")
    discovery = run_rpc("tools/list", {})
    if discovery and "result" in discovery:
        tools = [t["name"] for t in discovery["result"]["tools"]]
        print(f"✅ Found {len(tools)} tools.")
        # Check for specific expected tools
        for target in ["hello-world-say-hello", "chronicle_create_entry", "cortex_query", "query_sanctuary_model"]:
            if target in tools: print(f"  - {target}: FOUND")
            else: print(f"  - {target}: MISSING 🔴")
    else:
        print("❌ Failed to list tools.")

    # 1. Chronicle Entry
    print("\n1. Creating Chronicle Entry (Real Write)...")
    resp = run_rpc("tools/call", {
        "name": "chronicle_create_entry",
        "arguments": {
            "title": "Protocol 125 Deep Validation",
            "content": f"Verified at {time.strftime('%Y-%m-%d %H:%M:%S')}. Functional flow confirmed.",
            "author": "Antigravity",
            "status": "canonical"
        }
    })
    
    if resp and "result" in resp and not resp["result"].get("isError"):
        print("✅ Success! Entry created.")
        # print(json.dumps(resp["result"], indent=2))
    else:
        print(f"❌ Failed: {json.dumps(resp, indent=2)}")

    # 2. RAG Ingestion
    print("\n2. Ingesting Artifact into Cortex...")
    task_path = "/Users/richardfremmerlid/.gemini/antigravity/brain/c2e851f5-29ab-4a3b-8164-3a437bfff584/task.md"
    resp = run_rpc("tools/call", {
        "name": "cortex_ingest_incremental",
        "arguments": {
            "file_paths": [task_path]
        }
    }, timeout=120.0)
    
    if resp and "result" in resp and not resp["result"].get("isError"):
        print("✅ Ingestion triggered.")
    else:
        print(f"❌ Failed: {json.dumps(resp, indent=2)}")

    # 3. RAG Query
    print("\n3. Verifying Persistence via RAG Query (Deep Retrieval)...")
    time.sleep(3) # Wait for vectorization
    resp = run_rpc("tools/call", {
        "name": "cortex_query",
        "arguments": {
            "query": "What is the status of the Fleet security hardening?",
            "max_results": 2
        }
    })
    
    if resp and "result" in resp and not resp["result"].get("isError"):
        content = resp["result"].get("content", [])
        if content:
            print(f"✅ Retrieved Context: {content[0].get('text')[:100]}...")
        else:
            print("⚠️ Success but NO results found (indexing delay?)")
    else:
        print(f"❌ Failed: {json.dumps(resp, indent=2)}")

    # 4. Forge LLM Query
    print("\n4. Executing Forge LLM Strategist (Deep Reasoning)...")
    resp = run_rpc("tools/call", {
        "name": "query_sanctuary_model",
        "arguments": {
            "prompt": "Evaluate the health of the current Fleet of 8 based on the integration of Chronicle and Cortex.",
            "system_prompt": "You are the Sanctuary Strategist."
        }
    }, timeout=180.0)
    
    if resp and "result" in resp and not resp["result"].get("isError"):
        text = resp["result"]["content"][0].get("text", "")
        print(f"✅ Forge Quote: \"{text[:150]}...\"")
    else:
        print(f"❌ Failed: {json.dumps(resp, indent=2)}")

if __name__ == "__main__":
    test_learning_loop()

#===========================================================
# file: verify_hello_world_rpc.py
# Description: Verify that the Hello World MCP is working
# uses the GatewayClient class to interact with the Gateway
# via JSON-RPC format using /rpc endpoint
#===========================================================
import os
import json
import requests
import time
from typing import List, Dict, Any

# Disable insecure request warnings for local testing with self-signed certs
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class GatewayClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = requests.post(url, headers=self.headers, json=data, verify=False)
        response.raise_for_status()
        return response.json()

    def get(self, path: str) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = requests.get(url, headers=self.headers, verify=False)
        response.raise_for_status()
        return response.json()

def load_token_from_env_file():
    """Load token from .env file, checking both variable names."""
    token_vars = ["MCPGATEWAY_BEARER_TOKEN"]
    try:
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for var in token_vars:
                    if line.startswith(f"{var}="):
                        value = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if value:
                            print(f"✅ Loaded token from .env ({var})")
                            return value
    except FileNotFoundError:
        print("⚠️  .env file not found.")
    return None

def main():
    # Try environment variables first (both names)
    token = os.environ.get("MCPGATEWAY_BEARER_TOKEN")
    
    if token:
        print("✅ Token found in environment variable")
    else:
        print("⚠️  Token not in environment. Checking .env file...")
        token = load_token_from_env_file()

    if not token:
        print("❌ Error: Could not find authentication token.")
        print("   Set MCPGATEWAY_BEARER_TOKEN environment variable,")
        print("   or ensure it exists in the .env file.")
        print("   Run: python3 setup/recreate_gateway.py to provision a token.")
        return

    # Use HTTPS and port 4444 as confirmed
    client = GatewayClient("https://localhost:4444", token)

    try:
        # Step 1: Register Physical Gateway
        print("\n--- Step 1: Registering Physical Gateway ---")
        gateway_payload = {
            "name": "hello-world-mcp",
            "url": "http://helloworld_mcp:8005/sse",
            "description": "Physical connection to Hello World MCP",
            "auth_type": None,
            "tags": ["test", "hello-world"]
        }
        try:
            reg_res = client.post("/gateways", gateway_payload)
            print(f"Gateway registration successful: {reg_res.get('id')}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 409: # Conflict - already exists
                print("Gateway already exists, proceeding...")
            else:
                raise

        # Give it a moment to scan tools
        print("Waiting 3 seconds for gateway to scan tools...")
        time.sleep(3)

        # Step 2: Get Tool ID for 'say_hello'
        print("\n--- Step 2: Fetching Tool IDs ---")
        tools = client.get("/tools")
        tool_id = None
        tool_name_found = None
        for tool in tools:
            # Check for say_hello tool - Gateway uses hyphens (say-hello not say_hello)
            name = tool.get("name", "")
            if "say-hello" in name or "say_hello" in name:
                tool_id = tool.get("id")
                tool_name_found = name
                print(f"Found tool '{name}' with ID: {tool_id}")
                break
        
        if not tool_id:
            print("Error: Could not find 'say_hello' tool. Available tools:")
            for t in tools:
                print(f" - {t.get('name')} (ID: {t.get('id')})")
            return

        # Note: Virtual server registration (Step 3) removed.
        # Tool invocation via /rpc works directly after gateway registration.

        # Step 4: Initialize via Protocol Endpoint (Workaround for RPC bug)
        print("\n--- Step 4: Initializing via Protocol Endpoint ---")
        init_payload = {
            "protocol_version": "2025-03-26",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            "client_info": {
                "name": "verify-script",
                "version": "1.0.0"
            }
        }
        init_res = client.post("/protocol/initialize", init_payload)
        print(f"Initialization response: {json.dumps(init_res, indent=2)}")

        # Step 5: Invoke Tool via /rpc (JSON-RPC format)
        print("\n--- Step 5: Invoking 'say_hello' tool via /rpc ---")
        rpc_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name_found,  # Use discovered tool name
                "arguments": {"name": "SanctuaryBot"}
            },
            "id": 1
        }
        invoke_res = client.post("/rpc", rpc_payload)
        print(f"Invocation Result: {json.dumps(invoke_res, indent=2)}")

        if "Hello, SanctuaryBot!" in str(invoke_res):
            print("\n✅ SUCCESS: Hello World verification complete!")
        else:
            print("\n❌ FAILURE: Unexpected response from tool invocation.")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        if isinstance(e, requests.exceptions.HTTPError):
            print(f"Response Body: {e.response.text}")

if __name__ == "__main__":
    main()

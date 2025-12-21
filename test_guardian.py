import requests
import json
import os
from dotenv import load_dotenv

# Load explicitly from .env
load_dotenv()

# Configuration from .env
GATEWAY_URL = os.getenv("MCP_GATEWAY_URL", "https://localhost:4444")
BEARER_TOKEN = os.getenv("MCPGATEWAY_BEARER_TOKEN")
VERIFY_SSL = os.getenv("GATEWAY_VERIFY_SSL", "false").lower() == "true"

def call_tool(server, tool, arguments={}):
    url = f"{GATEWAY_URL}/rpc"
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "serverName": server,
        "toolName": tool,
        "arguments": arguments
    }
    
    # Standardize SSL warnings if verifying is False
    import urllib3
    if not VERIFY_SSL:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
    response = requests.post(url, headers=headers, json=payload, verify=VERIFY_SSL)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text, "status_code": response.status_code}

if __name__ == "__main__":
    print(f"Calling cortex-guardian-wakeup...")
    result = call_tool("sanctuary_cortex", "sanctuary_cortex-cortex-guardian-wakeup")
    print(json.dumps(result, indent=2))


import os
import sys
import asyncio
import httpx
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# tests/mcp_servers/gateway -> tests/mcp_servers -> tests -> root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway-verifier")

def load_env_file(filepath):
    """Simple env file loader"""
    if not os.path.exists(filepath):
        return {}
    env_vars = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip("'").strip('"')
    return env_vars

async def verify_gateway():
    # Load .env
    env_path = os.path.join(project_root, ".env")
    env = load_env_file(env_path)
    
    gateway_url = env.get("MCP_GATEWAY_URL", "https://localhost:4444")
    # Support both naming conventions
    token = env.get("MCPGATEWAY_BEARER_TOKEN") or env.get("MCP_GATEWAY_API_TOKEN")
    
    if not token:
        logger.error("❌ MCP_GATEWAY_API_TOKEN not found in .env")
        return

    logger.info(f"Target Gateway: {gateway_url}")
    logger.info("Token: present" if token else "Token: missing")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Payload for hello-world
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "hello-world-mcp-say-hello",
            "arguments": { "name": "SanctuaryAgent" }
        },
        "id": 1
    }

    try:
        async with httpx.AsyncClient(verify=False) as client:
            logger.info("Sending request...")
            response = await client.post(
                f"{gateway_url}/rpc",
                json=payload,
                headers=headers,
                timeout=10.0
            )
            
            logger.info(f"Response Status: {response.status_code}")
            if response.status_code == 200:
                logger.info("✅ SUCCESS!")
                logger.info(f"Response Body: {response.text}")
            else:
                logger.error(f"❌ FAILED. Status: {response.status_code}")
                logger.error(f"Response Body: {response.text}")

    except Exception as e:
        logger.error(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(verify_gateway())

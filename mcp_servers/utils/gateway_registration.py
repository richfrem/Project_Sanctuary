"""
Gateway Registration Module for sanctuary-utils
Implements Guardrail 2: Dynamic Self-Registration (ADR 060)

On container startup, registers with the MCP Gateway by POSTing
the tool manifest to the Gateway's registration endpoint.
"""
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("sanctuary-utils")

# Gateway configuration from environment
GATEWAY_URL = os.getenv("MCP_GATEWAY_URL", "https://localhost:4444")
GATEWAY_VERIFY_SSL = os.getenv("MCP_GATEWAY_VERIFY_SSL", "false").lower() == "true"
GATEWAY_API_TOKEN = os.getenv("MCP_GATEWAY_API_TOKEN", "")

# Container network alias (use Docker network when in compose)
CONTAINER_ENDPOINT = os.getenv(
    "SANCTUARY_UTILS_ENDPOINT", 
    "http://sanctuary-utils:8000"
)


async def register_with_gateway(manifest: dict[str, Any]) -> dict[str, Any]:
    """
    Register this container with the MCP Gateway.
    
    Args:
        manifest: Tool manifest containing server info and tools list
        
    Returns:
        Registration response from Gateway
    """
    registration_url = f"{GATEWAY_URL}/api/servers/register"
    
    # Build registration payload
    payload = {
        "server_name": manifest.get("server_name", "sanctuary-utils"),
        "endpoint": CONTAINER_ENDPOINT + "/sse",
        "health_check": CONTAINER_ENDPOINT + "/health",
        "version": manifest.get("version", "1.0.0"),
        "tools": manifest.get("tools", []),
    }
    
    headers = {
        "Content-Type": "application/json",
    }
    
    # Add API token if configured
    if GATEWAY_API_TOKEN:
        headers["Authorization"] = f"Bearer {GATEWAY_API_TOKEN}"
    
    try:
        async with httpx.AsyncClient(verify=GATEWAY_VERIFY_SSL) as client:
            response = await client.post(
                registration_url,
                json=payload,
                headers=headers,
                timeout=10.0,
            )
            
            if response.status_code in (200, 201):
                logger.info(f"✅ Registered with Gateway at {GATEWAY_URL}")
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "response": response.json() if response.content else {},
                }
            else:
                logger.warning(
                    f"⚠️ Gateway registration returned {response.status_code}: "
                    f"{response.text}"
                )
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text,
                }
                
    except httpx.ConnectError as e:
        logger.warning(f"⚠️ Gateway not available at {GATEWAY_URL}: {e}")
        return {
            "success": False,
            "error": f"Gateway not available: {e}",
            "recoverable": True,
        }
    except Exception as e:
        logger.error(f"❌ Gateway registration failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def deregister_from_gateway(server_name: str = "sanctuary-utils") -> dict[str, Any]:
    """
    Deregister this container from the MCP Gateway on shutdown.
    
    Args:
        server_name: Name of the server to deregister
        
    Returns:
        Deregistration response from Gateway
    """
    deregister_url = f"{GATEWAY_URL}/api/servers/{server_name}"
    
    headers = {}
    if GATEWAY_API_TOKEN:
        headers["Authorization"] = f"Bearer {GATEWAY_API_TOKEN}"
    
    try:
        async with httpx.AsyncClient(verify=GATEWAY_VERIFY_SSL) as client:
            response = await client.delete(
                deregister_url,
                headers=headers,
                timeout=5.0,
            )
            
            if response.status_code in (200, 204):
                logger.info(f"✅ Deregistered from Gateway")
                return {"success": True}
            else:
                logger.warning(f"⚠️ Deregistration returned {response.status_code}")
                return {"success": False, "status_code": response.status_code}
                
    except Exception as e:
        logger.warning(f"⚠️ Deregistration failed (non-critical): {e}")
        return {"success": False, "error": str(e)}

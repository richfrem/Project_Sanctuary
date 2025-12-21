#!/usr/bin/env python3
#=============================================================================
# FLEET ORCHESTRATOR: LAYER 3 (ORCHESTRATION & OBSERVATION)
#=============================================================================
# Drives the discovery loop and persists observed state to fleet_registry.json.
# Location: mcp_servers/gateway/fleet_orchestrator.py
#
# PRECONDITIONS:
#  1. Gateway must be reachable (MCP_GATEWAY_URL).
#  2. SSL/Auth tokens must be configured in environment.
#  3. Layer 2 (fleet_resolver.py) must be functional.
#  4. Fleet Clusters: Must be running via 'podman compose up' (docker-compose.yml).
#
# OUTPUTS:
#  1. fleet_registry.json (Discovery manifest with tool schemas).
#  2. Real-time logging of handshake status.
#
# QUICK REFERENCE:
#  1. run_discovery - Full registration + discovery + persistence loop.
#=============================================================================
import json
from pathlib import Path
from typing import Dict, Any

# Using local imports to avoid path issues during direct execution
try:
    from .fleet_resolver import get_resolved_fleet
    from .gateway_client import register_and_initialize, get_mcp_tools, wait_for_tools, GatewayConfig, get_session
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from mcp_servers.gateway.fleet_resolver import get_resolved_fleet
    from mcp_servers.gateway.gateway_client import register_and_initialize, get_mcp_tools, wait_for_tools, GatewayConfig, get_session

REGISTRY_PATH = Path(__file__).parent / "fleet_registry.json"

#=============================================================================
# 1. DISCOVERY ENGINE
#=============================================================================
# Purpose:  Iterate through resolved fleet, register with Gateway, 
#           initialize tool discovery, and capture responses.
# Output:   fleet_registry.json (Non-authoritative observation manifest).
#=============================================================================
def run_discovery():
    """Execute the full fleet registration and discovery loop."""
    config = GatewayConfig()
    resolved_fleet = get_resolved_fleet()
    results = {"fleet_servers": {}}
    
    print(f"🚀 Starting Fleet Discovery | Gateway: {config.url}")
    
    with get_session(config) as session:
        for alias, server in resolved_fleet.items():
            print(f"  ├─ {alias} ({server['slug']})...", end=" ", flush=True)
            
            # 1. Register and Initialize
            res = register_and_initialize(
                name=server["slug"],
                url=server["url"],
                description=server["description"],
                config=config,
                session=session
            )
            
            if not res["success"]:
                print(f"❌ FAILED: {res.get('error', 'unknown error')}")
                # Log the failure but continue to other servers
                results["fleet_servers"][alias] = {
                    **server,
                    "status": "error",
                    "error": res.get("error"),
                    "tools": []
                }
                continue
                
            # 2. Get Tools (with retry for SSE deferred discovery)
            print("handshake OK... waiting for tools...", end=" ", flush=True)
            tools_res = wait_for_tools(gateway_name=server["slug"], config=config, session=session, max_retries=5, retry_delay=2.0)
            
            if tools_res["success"]:
                print(f"found {tools_res['count']} tools. ✅")
                
                # Filter essential tool metadata to avoid JSON bloat
                filtered_tools = []
                for t in tools_res["tools"]:
                    filtered_tools.append({
                        "name": t.get("name"),
                        "description": t.get("description"),
                        "inputSchema": t.get("inputSchema")
                    })

                results["fleet_servers"][alias] = {
                    **server,
                    "status": "ready",
                    "tools": filtered_tools
                }
            else:
                print(f"tool discovery FAILED. ⚠️")
                results["fleet_servers"][alias] = {
                    **server,
                    "status": "partial",
                    "error": "tool discovery failed",
                    "tools": []
                }

    #=========================================================================
    # 2. PERSISTENCE
    #=========================================================================
    # Save the manifest to disk.
    #=========================================================================
    print(f"💾 Saving manifest to {REGISTRY_PATH}...")
    with open(REGISTRY_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("✨ Discovery Complete.")

if __name__ == "__main__":
    run_discovery()

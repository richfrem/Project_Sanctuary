#!/usr/bin/env python3
#=============================================================================
# GATEWAY CLIENT: FLEET-AWARE TRANSPORT LAYER
#=============================================================================
# This is a transport library for interacting with the Sanctuary Gateway.
# It is "Fleet-aware," meaning it can resolve cluster aliases (e.g., "git")
# using the fleet specification and resolution logic.
#
# PRECONDITIONS:
#  1. env_utils.py must be in scripts/ for environment loading.
#  2. fleet_spec and fleet_resolver must be present for alias support.
#  3. MCPGATEWAY_BEARER_TOKEN must be provided for RPC calls.
#  4. Gateway Infrastructure: Must be built/running via 'sanctuary-gateway' Makefile.
#
# CLI USAGE:
#  python -m mcp_servers.gateway.gateway_client <command> [options]
#
#  Commands:
#    pulse    - Test Gateway connectivity (default)
#    tools    - List all federated tools
#    servers  - List registered servers
#    register - Register a server from fleet spec
#    status   - Get server connection status
#
#  Options:
#    --server, -s   Server name to filter/operate on
#    --verbose, -v  Verbose output
#
# CLI EXAMPLES:
#  # Check Gateway connectivity
#  python -m mcp_servers.gateway.gateway_client pulse
#
#  # List all tools (verbose)
#  python -m mcp_servers.gateway.gateway_client tools -v
#
#  # List tools for sanctuary_git only
#  python -m mcp_servers.gateway.gateway_client tools --server sanctuary_git
#
#  # List registered servers
#  python -m mcp_servers.gateway.gateway_client servers
#
#  # Register sanctuary_git
#  python -m mcp_servers.gateway.gateway_client register --server git
#
#  # Get status of sanctuary_git
#  python -m mcp_servers.gateway.gateway_client status --server sanctuary_git
#
# PYTHON API:
#  from mcp_servers.gateway.gateway_client import get_mcp_tools, GatewayConfig
#  
#  config = GatewayConfig()
#  result = get_mcp_tools(gateway_name='sanctuary_git', config=config)
#  print(f"Tools: {result['count']}")
#
# QUICK REFERENCE:
#  1. register_mcp_server     - Physical registration (POST /gateways)
#  2. initialize_mcp_server   - Tool discovery (POST /protocol/initialize)
#  3. get_mcp_tools           - List tools (GET /tools, filter by gatewaySlug)
#  4. execute_mcp_tool        - Call a tool (POST /rpc)
#  5. register_and_initialize - Combined sequence of 1 & 2 (Alias-aware)
#  6. list_servers            - List all registered gateways (GET /gateways)
#  7. delete_server           - Remove a gateway (DELETE /gateways/{id})
#  8. clean_all_servers       - Remove all Sanctuary gateways
#=============================================================================
import os
import sys
import json
from pathlib import Path
from typing import Any, Optional, Dict
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from contextlib import asynccontextmanager

# Third-party
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from dotenv import load_dotenv

# Local imports
# Add project root to sys.path to find mcp_servers module if not installed
project_root = Path(__file__).resolve().parent.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Fleet logic with defensive try-except
try:
    from mcp_servers.lib.env_helper import get_env_variable
    from mcp_servers.gateway.fleet_spec import FleetSpec, ServerSpec, ToolSpec
    from mcp_servers.gateway.fleet_resolver import FleetResolver
except ImportError:
    # If not running as a package or files are missing, these remain None
    FLEET_SPEC = None
    resolve_server = None

@dataclass
class GatewayConfig:
    """Production configuration for Gateway interaction."""
    url: str = field(default_factory=lambda: get_env_variable("MCP_GATEWAY_URL", required=False) or "https://localhost:4444")
    token: str = field(default_factory=lambda: get_env_variable("MCPGATEWAY_BEARER_TOKEN", required=False) or "")
    verify_ssl: bool = field(default_factory=lambda: str(get_env_variable("GATEWAY_VERIFY_SSL", required=False)).lower() == "true")

#=============================================================================
# SESSION FACTORY: get_session
#=============================================================================
def get_session(config: Optional[GatewayConfig] = None) -> requests.Session:
    """Create a robust production requests session."""
    cfg = config or GatewayConfig()
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    
    session.headers.update({
        "Authorization": f"Bearer {cfg.token}",
        "User-Agent": "SanctuaryGatewayClient/1.5",
        "Connection": "close",
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    session.verify = cfg.verify_ssl
    return session


#=============================================================================
# 1. REGISTER MCP SERVER
#=============================================================================
def register_mcp_server(
    name: str, 
    url: str, 
    description: str = "", 
    transport: str = "SSE",
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """Register an MCP server with the Gateway."""
    config = config or GatewayConfig()
    inner_session = session or get_session(config)
    
    payload = {
        "name": name,
        "url": url,
        "description": description or f"Sanctuary {name}",
        "transport": transport
    }
    
    try:
        resp = inner_session.post(f"{config.url}/gateways", json=payload, timeout=15)
        
        if resp.status_code in [200, 201]:
            return {"success": True, "status": "registered", "data": resp.json()}
        elif resp.status_code == 409:
            return {"success": True, "status": "already_registered", "data": {}}
        else:
            return {"success": False, "status": "failed", "error": resp.text, "status_code": resp.status_code}
    except Exception as e:
        return {"success": False, "status": "error", "error": str(e)}
    finally:
        if not session:
            inner_session.close()


#=============================================================================
# 2. INITIALIZE MCP SERVER (Tool Discovery)
#=============================================================================
def initialize_mcp_server(
    gateway_name: str, 
    config: Optional[GatewayConfig] = None, 
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """Initialize an MCP server to trigger tool discovery."""
    config = config or GatewayConfig()
    inner_session = session or get_session(config)
    
    try:
        resp = inner_session.post(
            f"{config.url}/protocol/initialize",
            params={"gateway_name": gateway_name},
            json={"protocolVersion": "2024-11-05"},
            timeout=20
        )
        
        if resp.status_code == 200:
            return {"success": True, "status": "initialized", "data": resp.json() if resp.text else {}}
        else:
            error_msg = resp.text
            try:
                data = resp.json()
                if "detail" in data: error_msg = data["detail"]
                elif "message" in data: error_msg = data["message"]
            except: pass
            return {"success": False, "status": "failed", "error": error_msg, "status_code": resp.status_code}
    finally:
        if not session:
            inner_session.close()


#=============================================================================
# 3. DISCOVERY: get_mcp_tools
#=============================================================================
def get_gateway_status(
    gateway_name: str,
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """Get the connection status of a specific gateway."""
    config = config or GatewayConfig()
    inner_session = session or get_session(config)
    
    try:
        resp = inner_session.get(
            f"{config.url}/gateways/{gateway_name}",
            timeout=10
        )
        if resp.status_code != 200:
            return {"success": False, "error": resp.text, "status_code": resp.status_code}
        
        data = resp.json()
        return {
            "success": True,
            "connection_status": data.get("connection_status", "unknown"),
            "data": data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if not session:
            inner_session.close()


def get_mcp_tools(
    gateway_name: Optional[str] = None,
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None,
    per_page: int = 200
) -> Dict[str, Any]:
    """List tools from the Gateway using /admin/tools endpoint (Unified Registry).
    
    Note: Gateway converts underscores to hyphens in slugs.
    e.g., 'sanctuary_utils' becomes 'sanctuary-utils'
    
    Uses /admin/tools with pagination to fetch all tools (default limit was 50).
    """
    config = config or GatewayConfig()
    inner_session = session or get_session(config)
    
    try:
        # Use /admin/tools endpoint with pagination - fixes 50-item limit issue
        resp = inner_session.get(
            f"{config.url}/admin/tools",
            params={"per_page": per_page},
            timeout=15
        )
        
        if resp.status_code != 200:
            return {"success": False, "error": resp.text, "status_code": resp.status_code}
        
        data = resp.json()
        # Handle paginated response format: {data: [...], pagination: {...}}
        all_tools = data.get("data", []) if isinstance(data, dict) else data
        
        if gateway_name:
            # Convert underscores to hyphens for slug matching
            slug = gateway_name.replace("_", "-")
            all_tools = [t for t in all_tools if t.get("gatewaySlug") == slug]
        
        return {"success": True, "tools": all_tools, "count": len(all_tools)}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if not session:
            inner_session.close()


def wait_for_tools(
    gateway_name: str,
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None,
    max_retries: int = 5,
    retry_delay: float = 2.0
) -> Dict[str, Any]:
    """Wait for SSE tool discovery to complete with retry logic."""
    import time
    
    config = config or GatewayConfig()
    inner_session = session or get_session(config)
    
    try:
        for attempt in range(max_retries):
            # Check connection status first
            status_res = get_gateway_status(gateway_name, config, inner_session)
            connection = status_res.get("connection_status", "unknown")
            
            if connection == "connecting":
                # Still handshaking, wait and retry
                time.sleep(retry_delay)
                continue
            
            # Try to get tools
            tools_res = get_mcp_tools(gateway_name, config, inner_session)
            
            if tools_res.get("success") and tools_res.get("count", 0) > 0:
                return tools_res
            
            # No tools yet, wait and retry
            time.sleep(retry_delay)
        
        # Final attempt
        return get_mcp_tools(gateway_name, config, inner_session)
    finally:
        if not session:
            inner_session.close()


#=============================================================================
# 4. EXECUTE MCP TOOL
#=============================================================================
def execute_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Execute an MCP tool via the Gateway RPC endpoint."""
    config = config or GatewayConfig()
    inner_session = session or get_session(config)
    
    try:
        payload = {
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": 1,
        }
        resp = inner_session.post(f"{config.url}/rpc", json=payload, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            if "error" in data: return {"success": False, "error": data["error"]}
            return {"success": True, "result": data.get("result", {})}
        else:
            return {"success": False, "error": resp.text, "status_code": resp.status_code}
    finally:
        if not session:
            inner_session.close()


#=============================================================================
# 5. CONVENIENCE: Alias-Aware Flow
#=============================================================================
# Purpose: Support cluster aliases (e.g., "git") instead of full URLs.
#=============================================================================
def register_and_initialize(
    name: str,
    url: Optional[str] = None,
    description: str = "",
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Fleet-aware registration flow.
    If 'url' is None and 'name' matches a Fleet alias (e.g., "git"),
    the Resolver is used to find the canonical slug and URL.
    """
    registration_name = name
    registration_url = url
    registration_desc = description

    # Attempt alias resolution if url is missing and fleet logic is available
    if not registration_url and FLEET_SPEC and resolve_server:
        if name in FLEET_SPEC:
            resolved = resolve_server(FLEET_SPEC[name])
            registration_name = resolved["slug"]
            registration_url = resolved["url"]
            registration_desc = description or resolved["description"]
    
    if not registration_url:
        return {"success": False, "error": f"URL required for unknown server: {name}"}

    config = config or GatewayConfig()
    inner_session = session or get_session(config)
    
    try:
        # Step 1: Register
        reg_result = register_mcp_server(
            registration_name, registration_url, registration_desc, 
            config=config, session=inner_session
        )
        if not reg_result["success"]: return reg_result
        
        # Step 2: Initialize (tool discovery)
        init_result = initialize_mcp_server(registration_name, config=config, session=inner_session)
        
        return {
            "success": init_result["success"],
            "registration": reg_result,
            "initialization": init_result,
        }
    finally:
        if not session:
            inner_session.close()


#=============================================================================
# 6. ADMIN: List All Servers
#=============================================================================
def list_servers(
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """
    List all registered MCP servers from the Gateway.
    
    Returns:
        {"success": True, "servers": [...]} or {"success": False, "error": "..."}
    """
    cfg = config or GatewayConfig()
    inner_session = session or get_session(cfg)
    
    try:
        resp = inner_session.get(f"{cfg.url}/gateways", timeout=10)
        if resp.status_code == 200:
            return {"success": True, "servers": resp.json()}
        else:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if not session:
            inner_session.close()


#=============================================================================
# 7. ADMIN: Deactivate Server
#=============================================================================
def deactivate_server(
    server_id: str,
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """
    Deactivate an MCP server (disable without deleting).
    
    Args:
        server_id: The server ID from Gateway
        
    Returns:
        {"success": True} or {"success": False, "error": "..."}
    """
    cfg = config or GatewayConfig()
    inner_session = session or get_session(cfg)
    
    try:
        resp = inner_session.patch(
            f"{cfg.url}/gateways/{server_id}",
            json={"enabled": False},
            timeout=10
        )
        if resp.status_code in [200, 204]:
            return {"success": True}
        else:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if not session:
            inner_session.close()


#=============================================================================
# 8. ADMIN: Delete Server
#=============================================================================
def delete_server(
    server_id: str,
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """
    Permanently delete an MCP server from the Gateway.
    
    Args:
        server_id: The server ID from Gateway
        
    Returns:
        {"success": True} or {"success": False, "error": "..."}
    """
    cfg = config or GatewayConfig()
    inner_session = session or get_session(cfg)
    
    try:
        resp = inner_session.delete(
            f"{cfg.url}/gateways/{server_id}",
            timeout=10
        )
        if resp.status_code in [200, 204]:
            return {"success": True}
        else:
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if not session:
            inner_session.close()


#=============================================================================
# 9. ADMIN: Clean All Servers (Utility)
#=============================================================================
def clean_all_servers(
    config: Optional[GatewayConfig] = None,
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """
    Delete all registered MCP servers from the Gateway.
    Useful for clean slate before re-registration.
    
    Returns:
        {"success": True, "deleted_count": N} or {"success": False, "error": "..."}
    """
    cfg = config or GatewayConfig()
    inner_session = session or get_session(cfg)
    
    try:
        # Get all servers
        list_res = list_servers(config=cfg, session=inner_session)
        if not list_res["success"]:
            return list_res
        
        servers = list_res["servers"]
        deleted_count = 0
        
        for server in servers:
            server_id = server.get("id")
            if server_id:
                del_res = delete_server(server_id, config=cfg, session=inner_session)
                if del_res["success"]:
                    deleted_count += 1
                    print(f"  ✓ Deleted: {server.get('name', server_id)}")
                else:
                    print(f"  ✗ Failed to delete {server.get('name', server_id)}: {del_res.get('error')}")
        
        return {"success": True, "deleted_count": deleted_count}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if not session:
            inner_session.close()


#=============================================================================
# CLI INTERFACE
#=============================================================================
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Gateway Client CLI")
    parser.add_argument("command", nargs="?", default="pulse",
                       choices=["pulse", "tools", "servers", "register", "status", "execute"],
                       help="Command to run (default: pulse)")
    parser.add_argument("--server", "-s", type=str, default=None,
                       help="Server name to filter/operate on")
    parser.add_argument("--tool", "-t", type=str, default=None,
                       help="Tool name to execute")
    parser.add_argument("--args", "-a", type=str, default="{}",
                       help="Tool arguments as JSON string")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    config = GatewayConfig()
    
    print(f"📡 Gateway Client | {config.url}")
    print(f"   Command: {args.command}")
    if args.server:
        print(f"   Server: {args.server}")
    print()
    
    if args.command == "pulse":
        # Test gateway connectivity
        with get_session(config) as session:
            try:
                resp = session.get(f"{config.url}/tools", timeout=5)
                if resp.status_code == 200:
                    print("✅ Gateway Pulse: OK")
                else:
                    print(f"❌ Gateway Pulse: FAILED ({resp.status_code})")
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Connection: FAILED ({e})")
                sys.exit(1)

    elif args.command == "execute":
        # Test execute_mcp_tool
        if not args.tool:
            print("❌ --tool required for execute command")
            sys.exit(1)
            
        try:
            tool_args = json.loads(args.args)
        except json.JSONDecodeError:
            print("❌ Invalid JSON for --args")
            sys.exit(1)
            
        print(f"🚀 Executing {args.tool} with {tool_args}...")
        result = execute_mcp_tool(args.tool, tool_args, config=config)
        
        if result["success"]:
            print("✅ Result:")
            print(json.dumps(result["result"], indent=2))
        else:
            print(f"❌ Failed: {result.get('error')}")
            sys.exit(1)

    elif args.command == "tools":
        # Test get_mcp_tools
        result = get_mcp_tools(gateway_name=args.server, config=config)
        if result["success"]:
            print(f"✅ Total tools: {result['count']}")
            if args.server:
                print(f"   Filtered by: {args.server} → {args.server.replace('_', '-')}")
            
            # Group by gateway
            by_gateway = {}
            for t in result["tools"]:
                gw = t.get("gatewaySlug", "unknown")
                by_gateway.setdefault(gw, []).append(t)
            
            print("\n📦 Tools by Gateway:")
            for gw, tools in sorted(by_gateway.items()):
                print(f"   {gw}: {len(tools)} tools")
                if args.verbose:
                    for t in tools[:5]:
                        print(f"      - {t.get('name')}")
                    if len(tools) > 5:
                        print(f"      ... and {len(tools) - 5} more")
        else:
            print(f"❌ Failed: {result.get('error')}")
            sys.exit(1)

    elif args.command == "servers":
        # Test list_servers
        result = list_servers(config=config)
        if result["success"]:
            servers = result["servers"]
            print(f"✅ Registered servers: {len(servers)}")
            for s in servers:
                status = "🟢" if s.get("enabled") else "🔴"
                print(f"   {status} {s['name']} ({s.get('slug', 'N/A')})")
        else:
            print(f"❌ Failed: {result.get('error')}")
            sys.exit(1)

    elif args.command == "register":
        # Test register_and_initialize
        if not args.server:
            print("❌ --server required for register command")
            sys.exit(1)
        
        # Get server info from fleet spec
        try:
            from mcp_servers.gateway.fleet_resolver import get_resolved_fleet
            fleet = get_resolved_fleet()
            server_info = None
            for alias, spec in fleet.items():
                if alias == args.server or spec['slug'] == args.server:
                    server_info = spec
                    break
            
            if not server_info:
                print(f"❌ Server '{args.server}' not found in fleet spec")
                sys.exit(1)
            
            print(f"Registering {server_info['slug']}...")
            result = register_and_initialize(
                name=server_info['slug'],
                url=server_info['url'],
                description=server_info['description'],
                config=config
            )
            
            if result["success"]:
                print(f"✅ {server_info['slug']} registered successfully")
            else:
                print(f"❌ Failed: {result.get('error')}")
                sys.exit(1)
        except ImportError:
            print("❌ Cannot import fleet_resolver")
            sys.exit(1)

    elif args.command == "status":
        # Test get_gateway_status
        if not args.server:
            print("❌ --server required for status command")
            sys.exit(1)
        
        result = get_gateway_status(args.server, config=config)
        if result["success"]:
            print(f"✅ {args.server}")
            print(f"   Connection: {result.get('connection_status', 'unknown')}")
            data = result.get("data", {})
            print(f"   Enabled: {data.get('enabled', 'N/A')}")
            print(f"   Reachable: {data.get('reachable', 'N/A')}")
            print(f"   Last Seen: {data.get('lastSeen', 'N/A')}")
        else:
            print(f"❌ Failed: {result.get('error')}")
            sys.exit(1)



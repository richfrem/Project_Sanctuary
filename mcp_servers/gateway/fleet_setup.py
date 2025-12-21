#!/usr/bin/env python3
"""
Fleet Setup - Master Orchestration Script

This script orchestrates the complete Fleet lifecycle following the 3-Layer Architecture:
1. Spec Layer (fleet_spec.py) - Design Intent
2. Resolver Layer (fleet_resolver.py) - Policy Logic  
3. Execution Layer (gateway_client.py + fleet_orchestrator.py) - Transport

Workflow:
    1. Clean Gateway state (remove old registrations)
    2. Register all Fleet servers with Gateway
    3. Initialize tool discovery
    4. Persist observed state to fleet_registry.json
    5. Verify registration and tool counts

Usage:
    python3 -m mcp_servers.gateway.fleet_setup [--clean] [--verify]
    
    --clean   : Clean all existing servers before registration (default: True)
    --verify  : Run verification checks after registration (default: True)
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.gateway.gateway_client import (
    clean_all_servers, 
    list_servers,
    GatewayConfig
)
from mcp_servers.gateway.fleet_orchestrator import run_discovery
from mcp_servers.gateway.fleet_spec import FLEET_SPEC
import json

def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")

def verify_registration(config: GatewayConfig):
    """Verify that all expected servers are registered."""
    print_header("VERIFICATION")
    
    # Get registered servers
    list_res = list_servers(config=config)
    if not list_res["success"]:
        print(f"❌ Failed to list servers: {list_res.get('error')}")
        return False
    
    registered_servers = list_res["servers"]
    registered_names = {s.get("name") for s in registered_servers}
    
    # Get expected servers from spec
    expected_names = {spec.slug for spec in FLEET_SPEC.values()}
    
    # Compare
    print(f"Expected servers: {len(expected_names)}")
    print(f"Registered servers: {len(registered_names)}\n")
    
    missing = expected_names - registered_names
    extra = registered_names - expected_names
    
    if missing:
        print(f"❌ Missing servers: {missing}")
    
    if extra:
        print(f"⚠️  Extra servers: {extra}")
    
    if not missing and not extra:
        print("✅ All expected servers are registered!")
        
        # Check tool counts
        print("\nTool Discovery Status:")
        registry_path = Path(__file__).parent / "fleet_registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
            
            for alias, data in registry.get("fleet_servers", {}).items():
                tool_count = len(data.get("tools", []))
                status = data.get("status", "unknown")
                print(f"  • {alias:15} - {tool_count:2} tools ({status})")
        
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Fleet Setup - Master Orchestration")
    parser.add_argument("--clean", action="store_true", default=True,
                       help="Clean existing servers before registration")
    parser.add_argument("--no-clean", dest="clean", action="store_false",
                       help="Skip cleaning step")
    parser.add_argument("--verify", action="store_true", default=True,
                       help="Run verification after registration")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                       help="Skip verification step")
    parser.add_argument("--server", type=str, default=None,
                       help="Register only a specific server (e.g., 'sanctuary_git'). Default: all servers")
    
    args = parser.parse_args()
    config = GatewayConfig()
    
    print_header("FLEET SETUP - MASTER ORCHESTRATION")
    print(f"Gateway: {config.url}")
    print(f"Clean: {args.clean}")
    print(f"Verify: {args.verify}")
    print(f"Server: {args.server or 'ALL'}")
    
    # Step 1: Clean (if requested)
    if args.clean:
        print_header("STEP 1: CLEAN GATEWAY STATE")
        clean_res = clean_all_servers(config=config)
        
        if clean_res["success"]:
            print(f"✅ Cleaned {clean_res['deleted_count']} servers")
        else:
            print(f"❌ Clean failed: {clean_res.get('error')}")
            print("Continuing anyway...")
    
    # Step 2: Register & Discover
    print_header("STEP 2: REGISTER & DISCOVER")
    
    if args.server:
        # Register single server
        from mcp_servers.gateway.fleet_resolver import get_resolved_fleet
        from mcp_servers.gateway.gateway_client import register_and_initialize, get_mcp_tools
        
        resolved_fleet = get_resolved_fleet()
        
        # Find the server by alias or slug
        server_info = None
        for alias, spec in resolved_fleet.items():
            if alias == args.server or spec['slug'] == args.server:
                server_info = spec
                break
        
        if not server_info:
            print(f"❌ Server '{args.server}' not found in fleet spec")
            print(f"Available servers: {', '.join(resolved_fleet.keys())}")
            sys.exit(1)
        
        print(f"Registering {server_info['slug']}...")
        result = register_and_initialize(
            name=server_info['slug'],
            url=server_info['url'],
            description=server_info['description'],
            config=config
        )
        
        if result['success']:
            print(f"✅ {server_info['slug']} registered successfully")
        else:
            print(f"❌ {server_info['slug']} registration failed: {result.get('error')}")
            sys.exit(1)
    else:
        # Register all servers
        print("Running fleet orchestrator...")
        run_discovery()
    
    # Step 3: Verify (if requested)
    if args.verify:
        success = verify_registration(config)
        if not success:
            print("\n⚠️  Verification found issues - check output above")
            sys.exit(1)
    
    print_header("FLEET SETUP COMPLETE")
    print("✅ Fleet is registered and ready!")
    print(f"📄 Registry: mcp_servers/gateway/fleet_registry.json")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gateway Config Generator for Project Sanctuary.

This script helps generate the `claude_desktop_config.json` entry for the external
Sanctuary Gateway by inspecting the external directory and mapping local MCP servers.

Usage:
    python3 scripts/generate_gateway_config.py --gateway-path /path/to/sanctuary-gateway
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Fleet Configuration (Ports 8100-8105)
FLEET_CONFIG = {
    "mcpServers": {
        "utils": {
            "url": "http://localhost:8100/sse"
        },
        "filesystem": {
            "url": "http://localhost:8101/sse"
        },
        "network": {
            "url": "http://localhost:8102/sse"
        },
        "git": {
            "url": "http://localhost:8103/sse"
        },
        "cortex": {
            "url": "http://localhost:8104/sse"
        },
        "domain": {
            "url": "http://localhost:8105/sse"
        }
    }
}

def update_gateway_config(gateway_path: Path):
    """Updates the config.json inside the Gateway repo."""
    config_path = gateway_path / "config.json"
    
    try:
        with open(config_path, "w") as f:
            json.dump(FLEET_CONFIG, f, indent=2)
        logger.info(f"✅ Updated Gateway Config: {config_path}")
    except Exception as e:
        logger.error(f"Failed to write config.json: {e}")
        sys.exit(1)

def generate_claude_config(gateway_path: Path):
    """Generates the config JSON for Claude Desktop."""
    
    if not gateway_path.exists():
        logger.error(f"Gateway path does not exist: {gateway_path}")
        sys.exit(1)
        
    config = {
        "mcpServers": {
            "sanctuary-gateway": {
                "command": "podman",
                "args": [
                    "run",
                    "-i",
                    "--rm",
                    "--network", "host",
                    "-v", f"{gateway_path.resolve()}:/app",
                    "sanctuary-gateway",
                    "mcp-server-gateway"   
                ],
                "env": {
                    "GATEWAY_CONFIG_PATH": "/app/config.json"
                }
            }
        }
    }
    
    print("\nAdd this to your claude_desktop_config.json:")
    print(json.dumps(config, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Generate Sanctuary Gateway Config")
    parser.add_argument("--gateway-path", type=Path, required=True, help="Path to external sanctuary-gateway repo")
    
    args = parser.parse_args()
    
    # 1. Update the actual Gateway config file
    update_gateway_config(args.gateway_path)
    
    # 2. Output the Client config
    generate_claude_config(args.gateway_path)

if __name__ == "__main__":
    main()

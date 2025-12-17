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

def generate_config(gateway_path: Path):
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
                    "-v", f"{gateway_path}:/app",
                    "sanctuary-gateway",
                    "mcp-server-gateway"   
                ],
                "env": {
                    "GATEWAY_CONFIG_PATH": "/app/config.json"
                }
            }
        }
    }
    
    print(json.dumps(config, indent=2))
    logger.info("Config snippet generated above. Add to your claude_desktop_config.json")

def main():
    parser = argparse.ArgumentParser(description="Generate Sanctuary Gateway Config")
    parser.add_argument("--gateway-path", type=Path, required=True, help="Path to external sanctuary-gateway repo")
    
    args = parser.parse_args()
    generate_config(args.gateway_path)

if __name__ == "__main__":
    main()

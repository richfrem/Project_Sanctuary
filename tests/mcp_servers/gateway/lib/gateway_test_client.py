#!/usr/bin/env python3
#=============================================================================
# GATEWAY TEST CLIENT (MODULAR CORE)
#=============================================================================
# The canonical engine for all Tier 3 (Bridge) verification.
# Location: tests/mcp_servers/gateway/gateway_test_client.py
#
# QUICK REFERENCE:
#  1. call(tool, args)     - Execute an MCP tool via Gateway RPC
#  2. list_tools(slug)     - Discover available tools
#  3. health_check()       - Verify Gateway heartbeat
#
# DESIGN PHILOSOPHY:
#   Modular, reusable, and agnostic. This file serves as the single source 
#   of truth for how Sanctuary interacts with the Gateway RPC layer.
#=============================================================================
import os
import json
import requests
from pathlib import Path
from typing import Any, Optional, Dict, List
import pytest

#-----------------------------------------------------------------------------
# 0. ENVIRONMENT SETUP (Use Centralized Utility)
#-----------------------------------------------------------------------------
import sys
# Local imports
# Add project root to sys.path to find core modules
project_root = Path(__file__).resolve().parent.parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.gateway.gateway_client import (
    GatewayConfig,
    get_session,
    execute_mcp_tool,
    get_mcp_tools
)
from mcp_servers.gateway.fleet_resolver import get_resolved_fleet, resolve_server
from mcp_servers.gateway.fleet_spec import FLEET_SPEC

#=============================================================================
# THE ENGINE: GatewayTestClient (Test Harness)
#=============================================================================
class GatewayTestClient:
    """
    Tier 3 (Bridge) verification engine.
    
    This class wraps the production GatewayClient logic for use in test suites.
    It provides test-specific setup and convenience methods.
    """
    
    #=============================================================================
    # INITIALIZATION: __init__
    #=============================================================================
    # Purpose: Initialize the test harness using production config logic.
    #=============================================================================
    def __init__(self, config: Optional[GatewayConfig] = None):
        """Initialize the client. Uses production configuration logic."""
        self.config = config or GatewayConfig()
        self.session = get_session(self.config)

    #=============================================================================
    # 1. CORE RPC: call()
    #=============================================================================
    #=============================================================================
    # RPC CALL: call()
    #=============================================================================
    # Purpose: Wrap production execute_mcp_tool logic for testing.
    #=============================================================================
    def call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap production tool execution."""
        return execute_mcp_tool(tool_name, arguments, config=self.config, session=self.session)

    #=============================================================================
    # 2. DISCOVERY: list_tools()
    #=============================================================================
    #=============================================================================
    # DISCOVERY: list_tools()
    #=============================================================================
    # Purpose: Wrap production get_mcp_tools logic for discovery testing.
    #=============================================================================
    def list_tools(self, gateway_slug: Optional[str] = None) -> Dict[str, Any]:
        """Wrap production tool discovery."""
        return get_mcp_tools(gateway_slug, config=self.config, session=self.session)

    #=============================================================================
    # 3. HEARTBEAT: health_check()
    #=============================================================================
    #=============================================================================
    # HEARTBEAT: health_check()
    #=============================================================================
    # Purpose: Verify the Gateway /health endpoint is responsive.
    #=============================================================================
    def health_check(self) -> bool:
        """Verify the Gateway is alive."""
        try:
            resp = self.session.get(f"{self.config.url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    #=============================================================================
    # 5. FLEET RESOLUTION: get_fleet()
    #=============================================================================
    # Purpose: Provide tests with access to the resolved fleet topology.
    #=============================================================================
    def get_fleet(self) -> Dict[str, Dict[str, Any]]:
        """Get the current resolved fleet topology."""
        return get_resolved_fleet()

    def get_cluster_spec(self, alias: str) -> Dict[str, Any]:
        """Get resolution for a specific cluster alias."""
        if alias not in FLEET_SPEC:
            raise ValueError(f"Unknown cluster alias: {alias}")
        return resolve_server(FLEET_SPEC[alias])

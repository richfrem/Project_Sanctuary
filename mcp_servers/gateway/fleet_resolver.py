#!/usr/bin/env python3
#=============================================================================
# FLEET RESOLVER: LAYER 2 (POLICY LOGIC)
#=============================================================================
# Reconciles Design Intent (Spec) with the Runtime Context (Environment).
# Location: mcp_servers/gateway/fleet_resolver.py
#
# PRECONDITIONS:
#  1. Layer 1 (fleet_spec.py) must be accessible.
#  2. Environment variables (MCP_*_URL) should be set for overrides.
#
# OUTPUTS:
#  1. ResolvedServer dictionaries with final connection URLs.
#  2. Detailed source tracking (env vs spec).
#
# QUICK REFERENCE:
#  1. resolve_server      - Reconcile a single server spec.
#  2. get_resolved_fleet  - Resolve the entire fleet dictionary.
#=============================================================================
import os
from typing import Dict, Any
from .fleet_spec import FleetServerSpec, FLEET_SPEC

#=============================================================================
# 1. RESOLUTION LOGIC
#=============================================================================
# Purpose: Determine the actual URL for a cluster based on environment overrides.
# Pattern: MCP_<SLUG_UPPER>_URL (e.g., MCP_SANCTUARY_GIT_URL)
#=============================================================================
def resolve_server(spec: FleetServerSpec) -> Dict[str, Any]:
    """
    Resolve final runtime intent for a single server.
    Priority:
    1. Environment Variable (MCP_<SLUG_UPPER>_URL)
    2. Spec Default
    
    Returns a dictionary suitable for registration.
    """
    # Convert sanctuary_git to MCP_SANCTUARY_GIT_URL
    env_key = f"MCP_{spec.slug.replace('-', '_').upper()}_URL"
    resolved_url = os.getenv(env_key, spec.default_url)
    
    source = "env" if os.getenv(env_key) else "spec"
    
    return {
        "slug": spec.slug,
        "url": resolved_url,
        "description": spec.description,
        "required": spec.required,
        "source": source
    }

#=============================================================================
# 2. FLEET ORCHESTRATION HELPERS
#=============================================================================
# Purpose: Convenience functions for bulk resolution.
#=============================================================================
def get_resolved_fleet() -> Dict[str, Dict[str, Any]]:
    """
    Returns the entire fleet with fully resolved URLs.
    """
    resolved = {}
    for alias, spec in FLEET_SPEC.items():
        resolved[alias] = resolve_server(spec)
    return resolved

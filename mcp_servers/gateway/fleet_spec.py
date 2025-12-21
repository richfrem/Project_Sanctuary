#!/usr/bin/env python3
#=============================================================================
# FLEET SPECIFICATION: LAYER 1 (AUTHORITATIVE INTENT)
#=============================================================================
# This file defines the canonical "Design Intent" for the Sanctuary Fleet.
# Location: mcp_servers/gateway/fleet_spec.py
#
# PRECONDITIONS:
#  1. Correct cluster slugs must be known (sanctuary-*).
#  2. Default SSE network topology must be defined.
#
# OUTPUTS:
#  1. FLEET_SPEC: Global dictionary mapping aliases to FleetServerSpec.
#
# QUICK REFERENCE:
#  1. FleetServerSpec - Dataclass for cluster identity.
#  2. FLEET_SPEC      - Dictionary of all 6 clusters (Front-ends for 8 servers).
#=============================================================================
from dataclasses import dataclass
from typing import Dict

@dataclass
class FleetServerSpec:
    """Canonical identity for a server cluster in the Fleet of 8."""
    slug: str
    default_url: str
    description: str
    required: bool = True

#=============================================================================
# 1. THE FLEET SPEC (INTENT)
#=============================================================================
# Purpose: Authoritative mapping of clusters to their default SSE endpoints.
# Note:    These URLs are defaults and can be overridden via Resolver Layer.
#=============================================================================
FLEET_SPEC: Dict[str, FleetServerSpec] = {
    "utils": FleetServerSpec(
        slug="sanctuary_utils",
        default_url="http://sanctuary_utils:8000/sse",
        description="Low-risk, stateless tools (Time, Calc, UUID, String).",
        required=True
    ),
    "filesystem": FleetServerSpec(
        slug="sanctuary_filesystem",
        default_url="http://sanctuary_filesystem:8000/sse",
        description="High-risk file operations. Isolated from network.",
        required=True
    ),
    "network": FleetServerSpec(
        slug="sanctuary_network",
        default_url="http://sanctuary_network:8000/sse",
        description="External web access (Brave, Fetch). Isolated from filesystem.",
        required=True
    ),
    "git": FleetServerSpec(
        slug="sanctuary_git",
        default_url="http://sanctuary_git:8000/sse",
        description="Dual-permission (Filesystem + Network). Completely isolated container.",
        required=True
    ),
    "cortex": FleetServerSpec(
        slug="sanctuary_cortex",
        default_url="http://sanctuary_cortex:8000/sse",
        description="RAG, Forge LLM",
        required=True
    ),
    "domain": FleetServerSpec(
        slug="sanctuary_domain",
        default_url="http://sanctuary_domain:8105/sse",
        description="Chronicle, ADR, Protocol, Task",
        required=True
    ),
}

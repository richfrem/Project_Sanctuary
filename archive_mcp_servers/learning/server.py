#!/usr/bin/env python3
"""
Learning Server
=====================================

Purpose:
    Learning MCP Server (Protocol 128).
    Manages the learning loop, memory persistence (Soul), and cognitive snapshots.
    Handles semantic cache warming and system debriefs.

Layer: Interface (MCP)

Usage:
    # Run via MCP Config (STDIO)
    python -m mcp_servers.learning.server

    # Run via Gateway (SSE)
    PORT=8000 python -m mcp_servers.learning.server

Key Functions / MCP Tools:
    - learning_debrief(hours): Scan repo for changes (Scout)
    - capture_snapshot(manifest, type): Create audit/seal packet
    - persist_soul(snapshot, ...): Sync to Hugging Face (Incremental)
    - persist_soul_full(): Full sync to Hugging Face
    - guardian_wakeup(mode): Generate boot digest
    - guardian_snapshot(context): Create session start pack

Related:
    - mcp_servers/learning/operations.py
    - plugins/guardian-onboarding/resources/protocols/128_Hardened_Learning_Loop.md
"""
import sys
import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context
import mcp.types as types

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.learning.operations import LearningOperations
from mcp_servers.lib.sse_adaptor import SSEServer

# Setup logging
logger = setup_mcp_logging("learning")

# Initialize Operations
ops = LearningOperations(project_root)

# Initialize FastMCP
mcp = FastMCP("learning")

#=============================================================================
# TOOLS (Protocol 128 Lifecycle)
#=============================================================================

@mcp.tool(name="learning_debrief")
def learning_debrief(hours: int = 24) -> str:
    """
    Scans repository for technical state changes (Protocol 128).
    Returns a comprehensive markdown digest of recent activity.
    """
    return ops.learning_debrief(hours=hours)

@mcp.tool(name="capture_snapshot")
def capture_snapshot(
    manifest_files: list[str] = None, 
    snapshot_type: str = "audit",
    strategic_context: str = None
) -> dict:
    """
    Snapshot generation (Protocol 128). 
    Types: audit (red_team_audit_packet.md), seal (learning_package_snapshot.md), learning_audit (learning_audit_packet.md).
    """
    # Convert result to dict for JSON serialization
    res = ops.capture_snapshot(
        manifest_files=manifest_files,
        snapshot_type=snapshot_type,
        strategic_context=strategic_context
    )
    return {
        "status": res.status,
        "snapshot_path": str(res.snapshot_path) if res.snapshot_path else "",
        "total_files": res.total_files,
        "total_bytes": res.total_bytes,
        "manifest_verified": res.manifest_verified,
        "error": res.error
    }

@mcp.tool(name="persist_soul")
def persist_soul(
    snapshot_path: str,
    valence: float = 0.0,
    uncertainty: float = 0.0,
    is_full_sync: bool = False
) -> dict:
    """
    Incremental Soul persistence (ADR 079). 
    Uploads snapshot MD to lineage/ folder and appends 1 record to data/soul_traces.jsonl on HuggingFace.
    """
    from mcp_servers.learning.models import PersistSoulRequest
    
    req = PersistSoulRequest(
        snapshot_path=snapshot_path,
        valence=valence,
        uncertainty=uncertainty,
        is_full_sync=is_full_sync
    )
    res = ops.persist_soul(req)
    return {
        "status": res.status,
        "repo_url": res.repo_url,
        "snapshot_name": res.snapshot_name,
        "error": res.error
    }

@mcp.tool(name="persist_soul_full")
def persist_soul_full() -> dict:
    """
    Full Soul genome sync (ADR 081). 
    Regenerates data/soul_traces.jsonl from all project files (~1200 records) and deploys to HuggingFace.
    """
    res = ops.persist_soul_full()
    return {
        "status": res.status,
        "repo_url": res.repo_url,
        "snapshot_name": res.snapshot_name,
        "error": res.error
    }

@mcp.tool(name="guardian_wakeup")
def guardian_wakeup(mode: str = "HOLISTIC") -> dict:
    """
    Generate Guardian boot digest (Protocol 114).
    """
    res = ops.guardian_wakeup(mode=mode)
    return {
        "status": res.status,
        "digest_path": str(res.digest_path) if res.digest_path else "",
        "total_time_ms": res.total_time_ms,
        "error": res.error
    }

@mcp.tool(name="guardian_snapshot")
def guardian_snapshot(strategic_context: str = None) -> dict:
    """
    Captures the 'Guardian Start Pack' (Chronicle/Protocol/Roadmap) for session continuity.
    """
    res = ops.guardian_snapshot(strategic_context=strategic_context)
    return {
        "status": res.status,
        "snapshot_path": res.snapshot_path,
        "total_files": res.total_files,
        "total_bytes": res.total_bytes,
        "error": res.error
    }

#=============================================================================
# MAIN (Universal Entry Point)
#=============================================================================
if __name__ == "__main__":
    # Get Transport Mode from Environment
    transport = get_env_variable("MCP_TRANSPORT", required=False) or "stdio"
    port = int(get_env_variable("PORT", required=False) or "8000")

    logger.info(f"Starting Learning MCP Server (Transport: {transport})...")

    if transport.lower() == "sse":
        # Start SSE Server (Gateway Compatible)
        sse = SSEServer(mcp, host="0.0.0.0", port=port)
        sse.start()
    else:
        # Start Stdio Server (Local Dev / Claude)
        mcp.run()

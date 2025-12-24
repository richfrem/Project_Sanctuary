#============================================
# mcp_servers/rag_cortex/server.py
# Purpose: MCP Server for the Mnemonic Cortex RAG system.
#          Provides tools for ingestion, querying, and cache management.
# Role: Interface Layer
# Used as: Main service entry point.
#============================================

import os
import json
import sys
import logging
from typing import Optional, List, Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Configure environment to prevent stdout pollution - MUST BE FIRST
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.lib.container_manager import ensure_chromadb_running, ensure_ollama_running

from .models import (
    to_dict,
    CortexIngestFullRequest,
    CortexQueryRequest,
    CortexIngestIncrementalRequest,
    CortexCacheGetRequest,
    CortexCacheSetRequest,
    CortexCacheWarmupRequest,
    CortexGuardianWakeupRequest,
    CortexCaptureSnapshotRequest,
    CortexLearningDebriefRequest,
    ForgeQueryRequest
)

# 1. Initialize Logging
logger = setup_mcp_logging("project_sanctuary.rag_cortex")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.rag_cortex",
    instructions="""
    Mnemonic Cortex RAG Server.
    - Semantic search and ingestion of project knowledge.
    - Mnemonic Cache (CAG) management.
    - Snapshot capture and technical debriefing (Protocol 128).
    """
)

# 3. Global lazy instances
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
_cortex_ops = None
_forge_ops = None

def get_ops():
    global _cortex_ops
    if _cortex_ops is None:
        from .operations import CortexOperations
        _cortex_ops = CortexOperations(PROJECT_ROOT)
    return _cortex_ops

def get_forge_ops():
    global _forge_ops
    if _forge_ops is None:
        from mcp_servers.forge_llm.operations import ForgeOperations
        _forge_ops = ForgeOperations(PROJECT_ROOT)
    return _forge_ops

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def cortex_ingest_full(request: CortexIngestFullRequest) -> str:
    """Perform full re-ingestion of the knowledge base."""
    try:
        response = get_ops().ingest_full(
            purge_existing=request.purge_existing,
            source_directories=request.source_directories
        )
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_ingest_full: {e}")
        raise ToolError(f"Full ingestion failed: {str(e)}")

@mcp.tool()
async def cortex_query(request: CortexQueryRequest) -> str:
    """Perform semantic search query against the knowledge base."""
    try:
        response = get_ops().query(
            query=request.query,
            max_results=request.max_results,
            use_cache=request.use_cache,
            reasoning_mode=request.reasoning_mode
        )
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_query: {e}")
        raise ToolError(f"Query failed: {str(e)}")

@mcp.tool()
async def cortex_get_stats() -> str:
    """Get database statistics and health status."""
    try:
        response = get_ops().get_stats()
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_get_stats: {e}")
        raise ToolError(f"Stats retrieval failed: {str(e)}")

@mcp.tool()
async def cortex_ingest_incremental(request: CortexIngestIncrementalRequest) -> str:
    """Incrementally ingest documents."""
    try:
        response = get_ops().ingest_incremental(
            file_paths=request.file_paths,
            metadata=request.metadata,
            skip_duplicates=request.skip_duplicates
        )
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_ingest_incremental: {e}")
        raise ToolError(f"Incremental ingestion failed: {str(e)}")

@mcp.tool()
async def cortex_cache_get(request: CortexCacheGetRequest) -> str:
    """Retrieve cached answer for a query."""
    try:
        response = get_ops().cache_get(request.query)
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_cache_get: {e}")
        raise ToolError(f"Cache retrieval failed: {str(e)}")

@mcp.tool()
async def cortex_cache_set(request: CortexCacheSetRequest) -> str:
    """Store answer in cache."""
    try:
        response = get_ops().cache_set(request.query, request.answer)
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_cache_set: {e}")
        raise ToolError(f"Cache storage failed: {str(e)}")

@mcp.tool()
async def cortex_cache_warmup(request: CortexCacheWarmupRequest) -> str:
    """Pre-populate cache with genesis queries."""
    try:
        response = get_ops().cache_warmup(request.genesis_queries)
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_cache_warmup: {e}")
        raise ToolError(f"Cache warmup failed: {str(e)}")

@mcp.tool()
async def cortex_guardian_wakeup(request: CortexGuardianWakeupRequest) -> str:
    """Generate Guardian boot digest (Protocol 114)."""
    try:
        response = get_ops().guardian_wakeup(mode=request.mode)
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_guardian_wakeup: {e}")
        raise ToolError(f"Guardian wakeup failed: {str(e)}")

@mcp.tool()
async def cortex_cache_stats() -> str:
    """Get Mnemonic Cache (CAG) statistics."""
    try:
        stats = get_ops().get_cache_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_cache_stats: {e}")
        raise ToolError(f"Cache stats retrieval failed: {str(e)}")

@mcp.tool()
async def cortex_learning_debrief(request: CortexLearningDebriefRequest) -> str:
    """Scans repository for technical state changes (Protocol 128)."""
    try:
        response = get_ops().learning_debrief(hours=request.hours)
        return json.dumps({"status": "success", "debrief": response}, indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_learning_debrief: {e}")
        raise ToolError(f"Learning debrief failed: {str(e)}")

@mcp.tool()
async def cortex_capture_snapshot(request: CortexCaptureSnapshotRequest) -> str:
    """Tool-driven snapshot generation (Protocol 128 v3.5)."""
    try:
        response = get_ops().capture_snapshot(
            manifest_files=request.manifest_files,
            snapshot_type=request.snapshot_type,
            strategic_context=request.strategic_context
        )
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in cortex_capture_snapshot: {e}")
        raise ToolError(f"Snapshot capture failed: {str(e)}")

# Registry of Forge tools (mirrored for convenience in rag_cortex cluster)
@mcp.tool()
async def query_sanctuary_model(request: ForgeQueryRequest) -> str:
    """Query the fine-tuned Sanctuary model."""
    try:
        response = get_forge_ops().query_sanctuary_model(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt
        )
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        logger.error(f"Error in query_sanctuary_model: {e}")
        raise ToolError(f"Model query failed: {str(e)}")

@mcp.tool()
async def check_sanctuary_model_status() -> str:
    """Check Sanctuary model status."""
    try:
        result = get_forge_ops().check_model_availability()
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in check_sanctuary_model_status: {e}")
        raise ToolError(f"Status check failed: {str(e)}")

#============================================
# Main Execution Entry Point
#============================================

def run_server():
    # Ensure Containers are running
    if not get_env_variable("SKIP_CONTAINER_CHECKS", required=False):
        logger.info("Checking Container Services...")
        # 1. ChromaDB
        ensure_chromadb_running(PROJECT_ROOT)
        # 2. Ollama
        ensure_ollama_running(PROJECT_ROOT)

    # Dual-mode support
    port_env = get_env_variable("PORT", required=False)
    transport = "sse" if port_env else "stdio"
    
    if transport == "sse":
        port = int(port_env) if port_env else 8004
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)

if __name__ == "__main__":
    run_server()

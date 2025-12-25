#============================================
# mcp_servers/gateway/clusters/sanctuary_cortex/server.py
# Purpose: Sanctuary Cortex Cluster - Dual-Transport Entry Point
# Role: Interface Layer (Cluster Aggregator)
# Status: ADR-066 v1.3 Compliant (SSEServer for Gateway, FastMCP for STDIO)
# Used by: Gateway Fleet (SSE) and Claude Desktop (STDIO)
#============================================

import os
import sys
import json
import logging
from typing import Optional, List, Dict, Any

# Configure environment to prevent stdout pollution - MUST BE FIRST
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.lib.container_manager import ensure_chromadb_running, ensure_ollama_running

# Setup Logging
logger = setup_mcp_logging("project_sanctuary.sanctuary_cortex")

# Configuration
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
_cortex_ops = None
_forge_ops = None

def get_ops():
    global _cortex_ops
    if _cortex_ops is None:
        from mcp_servers.rag_cortex.operations import CortexOperations
        _cortex_ops = CortexOperations(PROJECT_ROOT)
    return _cortex_ops

def get_forge_ops():
    global _forge_ops
    if _forge_ops is None:
        from mcp_servers.forge_llm.operations import ForgeOperations
        _forge_ops = ForgeOperations(PROJECT_ROOT)
    return _forge_ops


#============================================
# Tool Schema Definitions (for SSEServer)
#============================================
INGEST_FULL_SCHEMA = {
    "type": "object",
    "properties": {
        "purge_existing": {"type": "boolean", "description": "Clear existing data first"},
        "source_directories": {"type": "array", "items": {"type": "string"}}
    }
}

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Semantic search query"},
        "max_results": {"type": "integer", "description": "Max results to return"},
        "use_cache": {"type": "boolean", "description": "Use cached results"},
        "reasoning_mode": {"type": "string", "description": "Reasoning mode"}
    },
    "required": ["query"]
}

INGEST_INCREMENTAL_SCHEMA = {
    "type": "object",
    "properties": {
        "file_paths": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
        "skip_duplicates": {"type": "boolean"}
    },
    "required": ["file_paths"]
}

CACHE_GET_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Query to look up"}
    },
    "required": ["query"]
}

CACHE_SET_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "answer": {"type": "string"}
    },
    "required": ["query", "answer"]
}

CACHE_WARMUP_SCHEMA = {
    "type": "object",
    "properties": {
        "genesis_queries": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["genesis_queries"]
}

GUARDIAN_WAKEUP_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "description": "full, fast, or minimal"}
    }
}

CAPTURE_SNAPSHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "manifest_files": {"type": "array", "items": {"type": "string"}},
        "snapshot_type": {"type": "string"},
        "strategic_context": {"type": "string"}
    }
}

LEARNING_DEBRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "hours": {"type": "integer", "description": "Hours to look back"}
    }
}

FORGE_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "temperature": {"type": "number"},
        "max_tokens": {"type": "integer"},
        "system_prompt": {"type": "string"}
    },
    "required": ["prompt"]
}

EMPTY_SCHEMA = {"type": "object", "properties": {}}

def to_dict(obj):
    """Convert response objects to dict recursively (handles nested dataclasses)."""
    # Handle Pydantic models
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    # Handle dataclasses (recursive)
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if isinstance(value, list):
                result[field_name] = [to_dict(item) for item in value]
            elif isinstance(value, dict):
                result[field_name] = {k: to_dict(v) for k, v in value.items()}
            else:
                result[field_name] = to_dict(value)
        return result
    # Handle plain dicts
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    # Handle lists
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    return obj


#============================================
# SSE Transport Implementation (Gateway Mode)
# Migrated to @sse_tool decorator pattern per ADR-076
#============================================
def run_sse_server(port: int):
    """Run using SSEServer for Gateway compatibility (ADR-066 v1.3)."""
    from mcp_servers.lib.sse_adaptor import SSEServer, sse_tool
    
    server = SSEServer("sanctuary_cortex", version="1.0.0")
    
    # =============================================================================
    # CORTEX INGESTION TOOLS (ADR-076 Decorator Pattern)
    # =============================================================================
    @sse_tool(
        name="cortex_ingest_full",
        description="Perform full re-ingestion of the knowledge base.",
        schema=INGEST_FULL_SCHEMA
    )
    def cortex_ingest_full(purge_existing: bool = False, source_directories: List[str] = None):
        response = get_ops().ingest_full(purge_existing=purge_existing, source_directories=source_directories)
        return json.dumps(to_dict(response), indent=2)
    
    @sse_tool(
        name="cortex_ingest_incremental",
        description="Incrementally ingest documents into the knowledge base.",
        schema=INGEST_INCREMENTAL_SCHEMA
    )
    def cortex_ingest_incremental(file_paths: List[str], metadata: Dict = None, skip_duplicates: bool = True):
        response = get_ops().ingest_incremental(file_paths=file_paths, metadata=metadata, skip_duplicates=skip_duplicates)
        return json.dumps(to_dict(response), indent=2)
    
    # =============================================================================
    # CORTEX QUERY TOOLS
    # =============================================================================
    @sse_tool(
        name="cortex_query",
        description="Perform semantic search query against the knowledge base.",
        schema=QUERY_SCHEMA
    )
    def cortex_query(query: str, max_results: int = 5, use_cache: bool = True, reasoning_mode: str = "standard"):
        response = get_ops().query(query=query, max_results=max_results, use_cache=use_cache, reasoning_mode=reasoning_mode)
        return json.dumps(to_dict(response), indent=2)
    
    @sse_tool(
        name="cortex_get_stats",
        description="Get database statistics and health status.",
        schema=EMPTY_SCHEMA
    )
    def cortex_get_stats():
        response = get_ops().get_stats()
        return json.dumps(to_dict(response), indent=2)
    
    # =============================================================================
    # CORTEX CACHE TOOLS
    # =============================================================================
    @sse_tool(
        name="cortex_cache_get",
        description="Retrieve cached answer for a query.",
        schema=CACHE_GET_SCHEMA
    )
    def cortex_cache_get(query: str):
        response = get_ops().cache_get(query)
        return json.dumps(to_dict(response), indent=2)
    
    @sse_tool(
        name="cortex_cache_set",
        description="Store answer in cache.",
        schema=CACHE_SET_SCHEMA
    )
    def cortex_cache_set(query: str, answer: str):
        response = get_ops().cache_set(query, answer)
        return json.dumps(to_dict(response), indent=2)
    
    @sse_tool(
        name="cortex_cache_warmup",
        description="Pre-populate cache with genesis queries.",
        schema=CACHE_WARMUP_SCHEMA
    )
    def cortex_cache_warmup(genesis_queries: List[str]):
        response = get_ops().cache_warmup(genesis_queries)
        return json.dumps(to_dict(response), indent=2)
    
    @sse_tool(
        name="cortex_cache_stats",
        description="Get Mnemonic Cache (CAG) statistics.",
        schema=EMPTY_SCHEMA
    )
    def cortex_cache_stats():
        stats = get_ops().get_cache_stats()
        return json.dumps(stats, indent=2)
    
    # =============================================================================
    # CORTEX PROTOCOL TOOLS (Guardian, Learning, Snapshot)
    # =============================================================================
    @sse_tool(
        name="cortex_guardian_wakeup",
        description="Generate Guardian boot digest (Protocol 114).",
        schema=GUARDIAN_WAKEUP_SCHEMA
    )
    def cortex_guardian_wakeup(mode: str = "full"):
        response = get_ops().guardian_wakeup(mode=mode)
        return json.dumps(to_dict(response), indent=2)
    
    @sse_tool(
        name="cortex_learning_debrief",
        description="Scans repository for technical state changes (Protocol 128).",
        schema=LEARNING_DEBRIEF_SCHEMA
    )
    def cortex_learning_debrief(hours: int = 24):
        response = get_ops().learning_debrief(hours=hours)
        return json.dumps({"status": "success", "debrief": response}, indent=2)
    
    @sse_tool(
        name="cortex_capture_snapshot",
        description="Tool-driven snapshot generation (Protocol 128 v3.5).",
        schema=CAPTURE_SNAPSHOT_SCHEMA
    )
    def cortex_capture_snapshot(manifest_files: List[str] = None, snapshot_type: str = "checkpoint", strategic_context: str = None):
        response = get_ops().capture_snapshot(manifest_files=manifest_files, snapshot_type=snapshot_type, strategic_context=strategic_context)
        return json.dumps(to_dict(response), indent=2)
    
    # =============================================================================
    # FORGE LLM TOOLS (Sanctuary Model)
    # =============================================================================
    @sse_tool(
        name="query_sanctuary_model",
        description="Query the fine-tuned Sanctuary model.",
        schema=FORGE_QUERY_SCHEMA
    )
    def query_sanctuary_model(prompt: str, temperature: float = 0.7, max_tokens: int = 2048, system_prompt: str = None):
        response = get_forge_ops().query_sanctuary_model(prompt=prompt, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt)
        return json.dumps(to_dict(response), indent=2)
    
    @sse_tool(
        name="check_sanctuary_model_status",
        description="Check Sanctuary model availability and status.",
        schema=EMPTY_SCHEMA
    )
    def check_sanctuary_model_status():
        result = get_forge_ops().check_model_availability()
        return json.dumps(result, indent=2)
    
    # Auto-register all decorated tools (ADR-076)
    server.register_decorated_tools(locals())
    
    logger.info(f"Starting SSEServer on port {port} (Gateway Mode)")
    server.run(port=port, transport="sse")


#============================================
# STDIO Transport Implementation (Local Mode)
#============================================
def run_stdio_server():
    """Run using FastMCP for local development (Claude Desktop)."""
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from mcp_servers.rag_cortex.models import (
        to_dict,
        CortexIngestFullRequest, CortexQueryRequest, CortexIngestIncrementalRequest,
        CortexCacheGetRequest, CortexCacheSetRequest, CortexCacheWarmupRequest,
        CortexGuardianWakeupRequest, CortexCaptureSnapshotRequest,
        CortexLearningDebriefRequest, ForgeQueryRequest
    )
    
    mcp = FastMCP(
        "sanctuary_cortex",
        instructions="""
        Sanctuary Cortex Cluster Aggregator.
        - specialized in semantic knowledge management (RAG).
        - specialized in specialized model reasoning (Forge).
        - handles technical state snapshots and learning debriefs.
        """
    )
    
    @mcp.tool()
    async def cortex_ingest_full(request: CortexIngestFullRequest) -> str:
        """Perform full re-ingestion of the knowledge base."""
        try:
            response = get_ops().ingest_full(purge_existing=request.purge_existing, source_directories=request.source_directories)
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Full ingestion failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_query(request: CortexQueryRequest) -> str:
        """Perform semantic search query against the knowledge base."""
        try:
            response = get_ops().query(query=request.query, max_results=request.max_results, use_cache=request.use_cache, reasoning_mode=request.reasoning_mode)
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Query failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_get_stats() -> str:
        """Get database statistics and health status."""
        try:
            response = get_ops().get_stats()
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Stats retrieval failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_ingest_incremental(request: CortexIngestIncrementalRequest) -> str:
        """Incrementally ingest documents."""
        try:
            response = get_ops().ingest_incremental(file_paths=request.file_paths, metadata=request.metadata, skip_duplicates=request.skip_duplicates)
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Incremental ingestion failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_cache_get(request: CortexCacheGetRequest) -> str:
        """Retrieve cached answer for a query."""
        try:
            response = get_ops().cache_get(request.query)
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Cache retrieval failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_cache_set(request: CortexCacheSetRequest) -> str:
        """Store answer in cache."""
        try:
            response = get_ops().cache_set(request.query, request.answer)
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Cache storage failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_cache_warmup(request: CortexCacheWarmupRequest) -> str:
        """Pre-populate cache with genesis queries."""
        try:
            response = get_ops().cache_warmup(request.genesis_queries)
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Cache warmup failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_guardian_wakeup(request: CortexGuardianWakeupRequest) -> str:
        """Generate Guardian boot digest (Protocol 114)."""
        try:
            response = get_ops().guardian_wakeup(mode=request.mode)
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Guardian wakeup failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_cache_stats() -> str:
        """Get Mnemonic Cache (CAG) statistics."""
        try:
            stats = get_ops().get_cache_stats()
            return json.dumps(stats, indent=2)
        except Exception as e:
            raise ToolError(f"Cache stats retrieval failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_learning_debrief(request: CortexLearningDebriefRequest) -> str:
        """Scans repository for technical state changes (Protocol 128)."""
        try:
            response = get_ops().learning_debrief(hours=request.hours)
            return json.dumps({"status": "success", "debrief": response}, indent=2)
        except Exception as e:
            raise ToolError(f"Learning debrief failed: {str(e)}")
    
    @mcp.tool()
    async def cortex_capture_snapshot(request: CortexCaptureSnapshotRequest) -> str:
        """Tool-driven snapshot generation (Protocol 128 v3.5)."""
        try:
            response = get_ops().capture_snapshot(manifest_files=request.manifest_files, snapshot_type=request.snapshot_type, strategic_context=request.strategic_context)
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Snapshot capture failed: {str(e)}")
    
    @mcp.tool()
    async def query_sanctuary_model(request: ForgeQueryRequest) -> str:
        """Query the fine-tuned Sanctuary model."""
        try:
            response = get_forge_ops().query_sanctuary_model(prompt=request.prompt, temperature=request.temperature, max_tokens=request.max_tokens, system_prompt=request.system_prompt)
            return json.dumps(to_dict(response), indent=2)
        except Exception as e:
            raise ToolError(f"Model query failed: {str(e)}")
    
    @mcp.tool()
    async def check_sanctuary_model_status() -> str:
        """Check Sanctuary model status."""
        try:
            result = get_forge_ops().check_model_availability()
            return json.dumps(result, indent=2)
        except Exception as e:
            raise ToolError(f"Status check failed: {str(e)}")
    
    logger.info("Starting FastMCP server (STDIO Mode)")
    mcp.run(transport="stdio")


#============================================
# Main Execution Entry Point (ADR-066 v1.3 Canonical Selector)
#============================================
def run_server():
    # Ensure Containers are running (optional check)
    if not get_env_variable("SKIP_CONTAINER_CHECKS", required=False):
        logger.info("Checking Container Services...")
        ensure_chromadb_running(PROJECT_ROOT)
        ensure_ollama_running(PROJECT_ROOT)
    
    MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio").lower()
    
    if MCP_TRANSPORT not in {"stdio", "sse"}:
        logger.error(f"Invalid MCP_TRANSPORT: {MCP_TRANSPORT}. Must be 'stdio' or 'sse'.")
        sys.exit(1)
    
    if MCP_TRANSPORT == "sse":
        port = int(os.getenv("PORT", 8000))
        run_sse_server(port)
    else:
        run_stdio_server()


if __name__ == "__main__":
    run_server()

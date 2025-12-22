"""
Cortex MCP Server
Domain: project_sanctuary.cognitive.cortex

Provides MCP tools for interacting with the Mnemonic Cortex RAG system.
Refactored to use SSEServer for Gateway integration (202 Accepted + Async SSE).
"""
import os
import json
import sys
import logging
from typing import Optional, List, Dict, Any

# Configure environment to prevent stdout pollution - MUST BE FIRST
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import SSEServer
# Ensure we can import from shared lib
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Utilities (Requested)
from mcp_servers.lib.utils.env_helper import get_env_variable
from mcp_servers.lib.utils.path_utils import find_project_root

try:
    from mcp_servers.lib.sse_adaptor import SSEServer
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib.sse_adaptor import SSEServer

from .validator import CortexValidator, ValidationError
from .models import to_dict
from mcp_servers.lib.container_manager import ensure_chromadb_running, ensure_ollama_running
from mcp_servers.forge_llm.operations import ForgeOperations
from mcp_servers.forge_llm.validator import ForgeValidator

# Initialize SSEServer
server = SSEServer("sanctuary_cortex")
app = server.app

# Global lazy instances
_cortex_ops = None
_cortex_validator = None
_forge_ops = None
_forge_validator = None
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()

def get_ops():
    global _cortex_ops
    if _cortex_ops is None:
        from .operations import CortexOperations
        _cortex_ops = CortexOperations(PROJECT_ROOT)
    return _cortex_ops

def get_validator():
    global _cortex_validator
    if _cortex_validator is None:
        _cortex_validator = CortexValidator(PROJECT_ROOT)
    return _cortex_validator

def get_forge_ops():
    global _forge_ops
    if _forge_ops is None:
        _forge_ops = ForgeOperations(PROJECT_ROOT)
    return _forge_ops

def get_forge_validator():
    global _forge_validator
    if _forge_validator is None:
        _forge_validator = ForgeValidator(PROJECT_ROOT)
    return _forge_validator

# Configure logging to write to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("rag_cortex")

# ----------------------------------------------------------------------
# Tool Handlers
# ----------------------------------------------------------------------

async def cortex_ingest_full(
    purge_existing: bool = True,
    source_directories: Optional[List[str]] = None
) -> str:
    """
    Perform full re-ingestion of the knowledge base.
    
    This operation purges the existing database and rebuilds it from scratch
    by processing all canonical documents. Use with caution.
    """
    try:
        # Validate inputs
        validated = get_validator().validate_ingest_full(
            purge_existing=purge_existing,
            source_directories=source_directories
        )
        
        # Perform ingestion
        response = get_ops().ingest_full(
            purge_existing=validated["purge_existing"],
            source_directories=validated["source_directories"]
        )
        
        # Convert to dict and return as JSON
        result = to_dict(response)
        return json.dumps(result, indent=2)
        
    except ValidationError as e:
        return json.dumps({"status": "error", "error": f"Validation error: {str(e)}"}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

async def cortex_query(
    query: str,
    max_results: int = 5,
    use_cache: bool = False,
    reasoning_mode: bool = False
) -> str:
    """
    Perform semantic search query against the knowledge base.
    
    Uses the Parent Document Retriever pattern to return full documents
    rather than fragmented chunks, providing complete context.
    """
    try:
        # Validate inputs
        validated = get_validator().validate_query(
            query=query,
            max_results=max_results,
            use_cache=use_cache
        )
        
        # Perform query
        # Currently synchronous op, running in async wrapper
        response = get_ops().query(
            query=validated["query"],
            max_results=validated["max_results"],
            use_cache=validated["use_cache"],
            reasoning_mode=reasoning_mode
        )
        
        # Convert to dict and return as JSON
        result = to_dict(response)
        return json.dumps(result, indent=2)
        
    except ValidationError as e:
        return json.dumps({"status": "error", "error": f"Validation error: {str(e)}"}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

async def cortex_get_stats() -> str:
    """Get database statistics and health status."""
    try:
        get_validator().validate_stats()
        response = get_ops().get_stats()
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

async def cortex_ingest_incremental(
    file_paths: List[str],
    metadata: Optional[dict] = None,
    skip_duplicates: bool = True
) -> str:
    """Incrementally ingest documents without rebuilding the entire database."""
    try:
        validated = get_validator().validate_ingest_incremental(
            file_paths=file_paths,
            metadata=metadata,
            skip_duplicates=skip_duplicates
        )
        
        response = get_ops().ingest_incremental(
            file_paths=validated["file_paths"],
            metadata=validated["metadata"],
            skip_duplicates=validated["skip_duplicates"]
        )
        
        result = to_dict(response)
        return json.dumps(result, indent=2)
        
    except ValidationError as e:
        return json.dumps({"status": "error", "error": f"Validation error: {str(e)}"}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

async def cortex_cache_get(query: str) -> str:
    """Retrieve cached answer for a query."""
    try:
        response = get_ops().cache_get(query)
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

async def cortex_cache_set(query: str, answer: str) -> str:
    """Store answer in cache for future retrieval."""
    try:
        response = get_ops().cache_set(query, answer)
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

async def cortex_cache_warmup(genesis_queries: Optional[List[str]] = None) -> str:
    """Pre-populate cache with genesis queries."""
    try:
        response = get_ops().cache_warmup(genesis_queries)
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

async def cortex_guardian_wakeup(mode: str = "HOLISTIC") -> str:
    """Generate Guardian boot digest from cached bundles (Protocol 114)."""
    try:
        response = get_ops().guardian_wakeup(mode=mode)
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

async def cortex_cache_stats() -> str:
    """Get Mnemonic Cache (CAG) statistics."""
    try:
        stats = get_ops().get_cache_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

# ----------------------------------------------------------------------
# Forge LLM Tools
# ----------------------------------------------------------------------

async def query_sanctuary_model(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None
) -> str:
    """Query the fine-tuned Sanctuary model for specialized knowledge."""
    try:
        validated = get_forge_validator().validate_query_sanctuary_model(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt
        )
        response = get_forge_ops().query_sanctuary_model(**validated)
        return json.dumps(to_dict(response), indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

async def check_sanctuary_model_status() -> str:
    """Check if the Sanctuary model is available and ready to use."""
    try:
        result = get_forge_ops().check_model_availability()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)

# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

# Schemas (Simplified for brevity, ideally mirrored from FastMCP introspection)
# We rely on the SSEServer to expose basic functionality

# Cortex Tools
server.register_tool("cortex_ingest_full", cortex_ingest_full, {
    "type": "object",
    "properties": {
        "purge_existing": {"type": "boolean", "default": True},
        "source_directories": {"type": "array", "items": {"type": "string"}}
    }
})
server.register_tool("cortex_query", cortex_query, {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "max_results": {"type": "integer", "default": 5},
        "use_cache": {"type": "boolean", "default": False},
        "reasoning_mode": {"type": "boolean", "default": False}
    },
    "required": ["query"]
})
server.register_tool("cortex_get_stats", cortex_get_stats, {"type": "object", "properties": {}})
server.register_tool("cortex_ingest_incremental", cortex_ingest_incremental, {
    "type": "object",
    "properties": {
        "file_paths": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
        "skip_duplicates": {"type": "boolean", "default": True}
    },
    "required": ["file_paths"]
})
server.register_tool("cortex_cache_get", cortex_cache_get, {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"]
})
server.register_tool("cortex_cache_set", cortex_cache_set, {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "answer": {"type": "string"}
    },
    "required": ["query", "answer"]
})
server.register_tool("cortex_cache_warmup", cortex_cache_warmup, {
    "type": "object",
    "properties": {"genesis_queries": {"type": "array", "items": {"type": "string"}}}
})
server.register_tool("cortex_guardian_wakeup", cortex_guardian_wakeup, {
    "type": "object",
    "properties": {"mode": {"type": "string", "default": "HOLISTIC"}}
})
server.register_tool("cortex_cache_stats", cortex_cache_stats, {"type": "object", "properties": {}})

# Forge Tools
server.register_tool("query_sanctuary_model", query_sanctuary_model, {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "temperature": {"type": "number", "default": 0.7},
        "max_tokens": {"type": "integer", "default": 2048},
        "system_prompt": {"type": "string"}
    },
    "required": ["prompt"]
})
server.register_tool("check_sanctuary_model_status", check_sanctuary_model_status, {"type": "object", "properties": {}})


if __name__ == "__main__":
    # Ensure Containers are running
    if not get_env_variable("SKIP_CONTAINER_CHECKS", required=False):
        logger.info("Checking Container Services...")

        # 1. ChromaDB
        success, message = ensure_chromadb_running(PROJECT_ROOT)
        if success:
            logger.info(f"✓ {message}")
        else:
            logger.error(f"✗ {message}")
            logger.warning("RAG operations may fail without ChromaDB")

        # 2. Ollama (for Embeddings/Reasoning)
        success, message = ensure_ollama_running(PROJECT_ROOT)
        if success:
            logger.info(f"✓ {message}")
        else:
            logger.error(f"✗ {message}")
            logger.warning("Embedding/Reasoning operations may fail without Ollama")
    else:
        logger.info("Skipping container checks (SKIP_CONTAINER_CHECKS set)")

    # Dual-mode support:
    # 1. If PORT is set -> Run as SSE (Gateway Mode)
    # 2. If PORT is NOT set -> Run as Stdio (Legacy Mode)
    port_env = get_env_variable("PORT", required=False)
    transport = "sse" if port_env else "stdio"
    port = int(port_env) if port_env else 8004
    
    server.run(port=port, transport=transport)

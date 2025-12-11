"""
Cortex MCP Server
Domain: project_sanctuary.cognitive.cortex

Provides MCP tools for interacting with the Mnemonic Cortex RAG system.
"""
import os
import json
import sys
import logging
from typing import Optional, List

# Configure environment to prevent stdout pollution - MUST BE FIRST
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastmcp import FastMCP
from .operations import CortexOperations
from .validator import CortexValidator, ValidationError
from .models import to_dict
from .container_manager import ensure_chromadb_running

# Initialize FastMCP with canonical domain name
mcp = FastMCP("project_sanctuary.cognitive.cortex")

# Initialize operations and validator
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
cortex_ops = CortexOperations(PROJECT_ROOT)
cortex_validator = CortexValidator(PROJECT_ROOT)

# Configure logging to write to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("rag_cortex")

# Ensure ChromaDB container is running
logger.info("Checking ChromaDB service...")
success, message = ensure_chromadb_running(PROJECT_ROOT)
if success:
    logger.info(f"✓ {message}")
else:
    logger.error(f"✗ {message}")
    logger.warning("Some operations may fail without ChromaDB service")



@mcp.tool()
def cortex_ingest_full(
    purge_existing: bool = True,
    source_directories: Optional[List[str]] = None
) -> str:
    """
    Perform full re-ingestion of the knowledge base.
    
    This operation purges the existing database and rebuilds it from scratch
    by processing all canonical documents. Use with caution.
    
    Args:
        purge_existing: Whether to purge existing database (default: True)
        source_directories: Optional list of source directories to ingest
                          (default: all canonical directories)
    
    Returns:
        JSON string with ingestion statistics
        
    Example:
        cortex_ingest_full()
        cortex_ingest_full(source_directories=["01_PROTOCOLS", "00_CHRONICLE"])
    """
    try:
        # Validate inputs
        validated = cortex_validator.validate_ingest_full(
            purge_existing=purge_existing,
            source_directories=source_directories
        )
        
        # Perform ingestion
        response = cortex_ops.ingest_full(
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


@mcp.tool()
def cortex_query(
    query: str,
    max_results: int = 5,
    use_cache: bool = False,
    reasoning_mode: bool = False
) -> str:
    """
    Perform semantic search query against the knowledge base.
    
    Uses the Parent Document Retriever pattern to return full documents
    rather than fragmented chunks, providing complete context.
    
    Args:
        query: Natural language query string
        max_results: Maximum number of results to return (default: 5, max: 100)
        use_cache: Whether to use cache (Phase 2 feature, default: False)
        reasoning_mode: Whether to use LLM to structure the query (default: False)
    
    Returns:
        JSON string with query results and metadata
        
    Example:
        cortex_query("What is Protocol 101?")
        cortex_query("Explain the Mnemonic Cortex architecture", max_results=3, reasoning_mode=True)
    """
    try:
        # Validate inputs
        # Note: We skip validation for reasoning_mode as it's a boolean
        validated = cortex_validator.validate_query(
            query=query,
            max_results=max_results,
            use_cache=use_cache
        )
        
        # Perform query
        response = cortex_ops.query(
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


@mcp.tool()
def cortex_get_stats() -> str:
    """
    Get database statistics and health status.
    
    Returns information about the number of documents, chunks, collections,
    and overall health of the RAG system.
    
    Returns:
        JSON string with database statistics
        
    Example:
        cortex_get_stats()
    """
    try:
        # Validate (no parameters needed)
        cortex_validator.validate_stats()
        
        # Get stats
        response = cortex_ops.get_stats()
        
        # Convert to dict and return as JSON
        result = to_dict(response)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


@mcp.tool()
def cortex_ingest_incremental(
    file_paths: List[str],
    metadata: Optional[dict] = None,
    skip_duplicates: bool = True
) -> str:
    """
    Incrementally ingest documents without rebuilding the entire database.
    
    This operation adds new documents to the existing knowledge base without
    purging existing data. Useful for adding new documents after initial ingestion.
    
    Args:
        file_paths: List of markdown file paths to ingest (absolute or relative)
        metadata: Optional metadata to attach to documents
        skip_duplicates: Whether to skip files already in database (default: True)
    
    Returns:
        JSON string with ingestion statistics
        
    Example:
        cortex_ingest_incremental(["00_CHRONICLE/2025-11-28_new_entry.md"])
        cortex_ingest_incremental(
            file_paths=["01_PROTOCOLS/120_new_protocol.md"],
            skip_duplicates=False
        )
    """
    try:
        # Validate inputs
        validated = cortex_validator.validate_ingest_incremental(
            file_paths=file_paths,
            metadata=metadata,
            skip_duplicates=skip_duplicates
        )
        
        # Perform incremental ingestion
        response = cortex_ops.ingest_incremental(
            file_paths=validated["file_paths"],
            metadata=validated["metadata"],
            skip_duplicates=validated["skip_duplicates"]
        )
        
        # Convert to dict and return as JSON
        result = to_dict(response)
        return json.dumps(result, indent=2)
        
    except ValidationError as e:
        return json.dumps({"status": "error", "error": f"Validation error: {str(e)}"}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


# ============================================================================
# Cache Operations (Protocol 114 - Guardian Wakeup)
# ============================================================================

@mcp.tool()
def cortex_cache_get(query: str) -> str:
    """
    Retrieve cached answer for a query.
    
    Checks the Mnemonic Cache (CAG) for a previously computed answer.
    Returns cache hit status and answer if found.
    
    Args:
        query: Query string to look up in cache
    
    Returns:
        JSON with cache hit status and answer if found
    
    Example:
        cortex_cache_get("What is Protocol 101?")
    """
    try:
        response = cortex_ops.cache_get(query)
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


@mcp.tool()
def cortex_cache_set(query: str, answer: str) -> str:
    """
    Store answer in cache for future retrieval.
    
    Caches an answer for a specific query in the Mnemonic Cache (CAG).
    Subsequent identical queries will retrieve this cached answer instantly.
    
    Args:
        query: Query string (cache key)
        answer: Answer to cache
    
    Returns:
        JSON with cache storage confirmation
    
    Example:
        cortex_cache_set("What is Protocol 101?", "Protocol 101 is...")
    """
    try:
        response = cortex_ops.cache_set(query, answer)
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


@mcp.tool()
def cortex_cache_warmup(genesis_queries: Optional[List[str]] = None) -> str:
    """
    Pre-populate cache with genesis queries.
    
    Warms up the cache by pre-computing answers for frequently asked questions.
    If no queries provided, uses default set of essential Sanctuary questions.
    
    Args:
        genesis_queries: Optional list of queries to cache. If None, uses defaults.
    
    Returns:
        JSON with warmup statistics (queries cached, cache hits/misses, time)
    
    Example:
        cortex_cache_warmup()
        cortex_cache_warmup(genesis_queries=["What is Protocol 87?", "Latest roadmap"])
    """
    try:
        response = cortex_ops.cache_warmup(genesis_queries)
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


@mcp.tool()
def cortex_guardian_wakeup(mode: str = "HOLISTIC") -> str:
    """
    Generate Guardian boot digest from cached bundles (Protocol 114).
    
    Retrieves chronicles, protocols, and roadmap summaries from cache
    and writes a digest to WORK_IN_PROGRESS/guardian_boot_digest.md.
    This provides the Guardian with essential context on startup.
    
    Returns:
        JSON with digest path and cache statistics
    
    Example:
        cortex_guardian_wakeup()
    """
    try:
        response = cortex_ops.guardian_wakeup(mode=mode)
        result = to_dict(response)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


@mcp.tool()
def cortex_cache_stats() -> str:
    """
    Get Mnemonic Cache (CAG) statistics.
    
    Returns information about hot/warm cache size and hit rates.
    
    Returns:
        JSON with cache statistics
        
    Example:
        cortex_cache_stats()
    """
    try:
        stats = cortex_ops.get_cache_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


# Legacy import - commented out as mnemonic_cortex module was archived
# from mnemonic_cortex.app.synthesis.generator import SynthesisGenerator

# @mcp.tool()
# def cortex_generate_adaptation_packet(days: int = 7) -> str:
#     """
#     Synthesize recent Cortex knowledge into an Adaptation Packet for model fine-tuning.
#     
#     Args:
#         days: Number of days to look back for changes (default: 7)
#         
#     Returns:
#         Path to the generated packet file.
#     """
#     generator = SynthesisGenerator(PROJECT_ROOT)
#     packet = generator.generate_packet(days=days)
#     output_path = generator.save_packet(packet)
#     return f"Generated Adaptation Packet: {output_path}"

if __name__ == "__main__":
    mcp.run()

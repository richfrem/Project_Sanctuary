#!/usr/bin/env python3
"""
Test script for RAG Cortex Full Ingestion.
Uses CortexOperations directly to verify the local HuggingFace architecture.
"""
import os
import sys
import time
import logging
from pathlib import Path

# Add project root based on .git marker (Robust)
current = Path(__file__).resolve().parent
while not (current / ".git").exists():
    if current == current.parent:
        raise RuntimeError("Could not find Project_Sanctuary root (no .git folder found)")
    current = current.parent
project_root = current
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.lib.utils.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging

# Configure logging using standard util
logger = setup_mcp_logging("test_full_ingest", log_file="logs/test_ingest.log")

def run_ingest():
    try:
        root = find_project_root()
        logger.info(f"Using Project Root: {root}")
        
        # Initialize operations (Uses HuggingFaceEmbeddings as per ADR 069)
        ops = CortexOperations(root)
        
        logger.info("Starting Full Ingestion (purge_existing=True)...")
        start_time = time.time()
        
        # Perform full ingest
        response = ops.ingest_full(purge_existing=True)
        
        elapsed = time.time() - start_time
        
        if response.status == "success":
            logger.info("✓ Full Ingestion Successful!")
            logger.info(f"  - Documents Processed: {response.documents_processed}")
            logger.info(f"  - Chunks Created: {response.chunks_created}")
            logger.info(f"  - Time Elapsed: {elapsed:.2f}s")
            logger.info(f"  - VectorStore: {response.vectorstore_path}")
        else:
            logger.error(f"✗ Ingestion Failed: {response.error}")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)

if __name__ == "__main__":
    run_ingest()

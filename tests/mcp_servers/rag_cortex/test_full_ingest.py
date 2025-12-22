#!/usr/bin/env python3
"""
Test script for RAG Cortex Full Ingestion.
Uses CortexOperations directly to verify the local Nomic architecture.
"""
import os
import sys
import time
import logging
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.lib.utils.path_utils import find_project_root

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_full_ingest")

def run_ingest():
    try:
        root = find_project_root()
        logger.info(f"Using Project Root: {root}")
        
        # Initialize operations with local Nomic mode
        ops = CortexOperations(root)
        
        logger.info("Starting Full Ingestion (purge_existing=True)...")
        start_time = time.time()
        
        # Use Nomic for all (this is enforced in CortexOperations.__init__)
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

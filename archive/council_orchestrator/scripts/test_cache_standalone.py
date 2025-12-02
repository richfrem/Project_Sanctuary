# council_orchestrator/scripts/test_cache_standalone.py
# Standalone cache verification script - tests cache functionality without orchestrator

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from council_orchestrator.orchestrator.memory.cache import CacheManager
from council_orchestrator.orchestrator.memory.cortex import CortexManager
from council_orchestrator.orchestrator.handlers.cache_wakeup_handler import render_guardian_boot_digest

def setup_logging():
    """Set up logging for the standalone test."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    return logging.getLogger(__name__)

def test_cache_prefill(cache_manager, logger):
    """Test cache prefill from RAG DB."""
    logger.info("Testing cache prefill from RAG DB...")
    try:
        cache_manager.prefill_guardian_start_pack()
        logger.info("Cache prefill completed successfully")
        return True
    except Exception as e:
        logger.error(f"Cache prefill failed: {e}")
        return False

def test_digest_generation(cache_manager, output_path, logger):
    """Test digest generation from cache."""
    logger.info("Testing digest generation from cache...")
    try:
        # Fetch data from cache
        result = cache_manager.fetch_guardian_start_pack(
            bundles=["chronicles", "protocols", "roadmap"],
            limit=15
        )

        # Render digest
        digest_content = render_guardian_boot_digest(result, project_root)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(digest_content)

        logger.info(f"Digest generated successfully: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Digest generation failed: {e}")
        return False

def verify_outputs(output_path, logger):
    """Verify that outputs were created correctly."""
    logger.info("Verifying outputs...")

    if not output_path.exists():
        logger.error(f"Output file not created: {output_path}")
        return False

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if "# Guardian Boot Digest (Cache)" not in content:
            logger.error("Digest file missing expected header")
            return False

        logger.info("Output verification successful")
        return True
    except Exception as e:
        logger.error(f"Error reading output file: {e}")
        return False

def main():
    """Main test function."""
    logger = setup_logging()
    logger.info("Starting standalone cache verification...")

    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "WORK_IN_PROGRESS" / "guardian_boot_digest.md"

    # Create managers
    try:
        # Create a mock logger for the managers
        mock_logger = logging.getLogger("cache_test")

        # Create CacheManager
        cache_manager = CacheManager(project_root, mock_logger)

        # Create CortexManager (needed for cache_manager initialization)
        cortex_manager = CortexManager(project_root, mock_logger)
        cortex_manager.cache_manager = cache_manager

    except Exception as e:
        logger.error(f"Failed to initialize managers: {e}")
        return False

    # Run tests
    success = True

    # Test 1: Cache prefill
    if not test_cache_prefill(cache_manager, logger):
        success = False

    # Test 2: Digest generation
    if not test_digest_generation(cache_manager, output_path, logger):
        success = False

    # Test 3: Verify outputs
    if not verify_outputs(output_path, logger):
        success = False

    # Final result
    if success:
        logger.info("Cache verification complete - All tests passed!")
        logger.info(f"Check the digest file: {output_path}")
    else:
        logger.error("Cache verification failed - Check logs above")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Integration tests for Cortex MCP Server

Tests all 4 tools in order of speed:
1. cortex_get_stats (fastest)
2. cortex_query (fast)
3. cortex_ingest_incremental (medium)
4. cortex_ingest_full (slowest - optional)

Usage:
    python3 run_cortex_integration.py
    python3 run_cortex_integration.py --run-full-ingest
"""
import sys
import json
import time
import tempfile
import argparse
from pathlib import Path

# Add project root to path
# tests/mcp_servers/rag_cortex/run_cortex_integration.py -> tests/mcp_servers -> tests -> Project_Sanctuary
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from the parent package
from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.rag_cortex.validator import CortexValidator
from mcp_servers.rag_cortex.models import to_dict
from mcp_servers.lib.utils.env_helper import get_env_variable, load_env


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_test_header(test_name: str):
    """Print test header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}TEST: {test_name}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.YELLOW}ℹ {message}{Colors.RESET}")


def test_cortex_get_stats(ops: CortexOperations) -> bool:
    """Test cortex_get_stats tool."""
    print_test_header("cortex_get_stats")
    
    try:
        start = time.time()
        response = ops.get_stats()
        elapsed = time.time() - start
        
        result = to_dict(response)
        
        # Validate response (StatsResponse doesn't have 'status', only 'health_status')
        assert 'error' not in result or result['error'] is None, f"Got error: {result.get('error')}"
        assert result['health_status'] in ['healthy', 'degraded', 'error'], f"Invalid health status: {result['health_status']}"
        assert 'total_documents' in result, "Missing total_documents"
        assert 'total_chunks' in result, "Missing total_chunks"
        assert 'collections' in result, "Missing collections"
        
        print_success(f"Stats retrieved in {elapsed:.2f}s")
        print_info(f"Health: {result['health_status']}")
        print_info(f"Documents: {result['total_documents']}")
        print_info(f"Chunks: {result['total_chunks']}")
        
        if result['health_status'] == 'healthy':
            print_success("Database is healthy")
            return True
        else:
            print_error(f"Database health is {result['health_status']}")
            return False
            
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def test_cortex_query(ops: CortexOperations) -> bool:
    """Test cortex_query tool."""
    print_test_header("cortex_query")
    
    test_queries = [
        ("What is Protocol 101?", 3),
        ("Covenant of Grace chronicle entry", 2),
        ("Mnemonic Cortex architecture", 2)
    ]
    
    all_passed = True
    
    for query, max_results in test_queries:
        try:
            print_info(f"Query: '{query}' (max_results={max_results})")
            
            start = time.time()
            response = ops.query(query, max_results=max_results)
            elapsed = time.time() - start
            
            result = to_dict(response)
            
            # Validate response
            assert result['status'] == 'success', f"Expected success, got {result['status']}"
            assert 'results' in result, "Missing results"
            assert 'query_time_ms' in result, "Missing query_time_ms"
            assert len(result['results']) <= max_results, f"Too many results: {len(result['results'])}"
            
            print_success(f"Query completed in {elapsed:.2f}s")
            print_info(f"Results: {len(result['results'])} documents")
            
            # Show first result preview
            if result['results']:
                first_result = result['results'][0]
                content_preview = first_result['content'][:150].replace('\n', ' ')
                print_info(f"First result: {content_preview}...")
            
        except Exception as e:
            print_error(f"Query failed: {str(e)}")
            all_passed = False
    
    return all_passed


def test_cortex_ingest_incremental(ops: CortexOperations) -> bool:
    """Test cortex_ingest_incremental tool."""
    print_test_header("cortex_ingest_incremental")
    
    try:
        # Create a temporary test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            test_content = f"""# Test Document for Cortex MCP Integration

**Date:** {time.strftime('%Y-%m-%d')}
**Type:** Integration Test

## Purpose

This document is created automatically by the Cortex MCP integration test suite
to verify that incremental ingestion works correctly.

## Test Data

- Test ID: cortex_mcp_integration_test_{int(time.time())}
- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Purpose: Verify cortex_ingest_incremental functionality

## Expected Behavior

This document should be:
1. Successfully ingested into the Mnemonic Cortex
2. Searchable via cortex_query
3. Retrievable with full content intact

## Cleanup

This test document can be safely removed after testing.
"""
            f.write(test_content)
            test_file = f.name
        
        print_info(f"Created test document: {test_file}")
        
        # Test ingestion
        start = time.time()
        response = ops.ingest_incremental(
            file_paths=[test_file],
            skip_duplicates=True
        )
        elapsed = time.time() - start
        
        result = to_dict(response)
        
        # Validate response
        assert result['status'] == 'success', f"Expected success, got {result['status']}: {result.get('error', '')}"
        assert 'documents_added' in result, "Missing documents_added"
        assert 'chunks_created' in result, "Missing chunks_created"
        
        print_success(f"Incremental ingest completed in {elapsed:.2f}s")
        print_info(f"Documents added: {result['documents_added']}")
        print_info(f"Chunks created: {result['chunks_created']}")
        print_info(f"Skipped duplicates: {result['skipped_duplicates']}")
        
        # Verify document is searchable
        print_info("Verifying document is searchable...")
        query_response = ops.query("cortex_mcp_integration_test", max_results=1)
        query_result = to_dict(query_response)
        
        if query_result['status'] == 'success' and len(query_result['results']) > 0:
            print_success("Document is searchable via cortex_query")
        else:
            print_error("Document not found in search results")
            return False
        
        # Cleanup
        Path(test_file).unlink()
        print_info(f"Cleaned up test document: {test_file}")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        # Cleanup on error
        try:
            if 'test_file' in locals():
                Path(test_file).unlink()
        except:
            pass
        return False


def test_cortex_ingest_full(ops: CortexOperations) -> bool:
    """Test cortex_ingest_full tool (SLOW - optional)."""
    print_test_header("cortex_ingest_full (SLOW)")
    
    print_info("This test performs a full database re-ingestion")
    print_info("It may take several minutes to complete")
    
    try:
        start = time.time()
        response = ops.ingest_full(purge_existing=True)
        elapsed = time.time() - start
        
        result = to_dict(response)
        
        # Validate response
        assert result['status'] == 'success', f"Expected success, got {result['status']}: {result.get('error', '')}"
        assert 'documents_processed' in result, "Missing documents_processed"
        assert 'ingestion_time_ms' in result, "Missing ingestion_time_ms"
        
        print_success(f"Full ingest completed in {elapsed:.2f}s")
        print_info(f"Documents processed: {result['documents_processed']}")
        print_info(f"Vectorstore: {result['vectorstore_path']}")
        
        return True
        
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False


def main():
    """Run all integration tests."""
    parser = argparse.ArgumentParser(description='Cortex MCP Integration Tests')
    parser.add_argument('--run-full-ingest', action='store_true',
                       help='Run the slow full ingestion test')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only run the stats and health check')
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{'='*60}")
    print("Cortex MCP Server - Integration Test Suite")
    print(f"{'='*60}{Colors.RESET}\n")
    
    # Load environment variables
    load_env()
    print_info("Loaded environment via env_helper")
    
    # Initialize operations
    ops = CortexOperations(str(project_root))
    
    # Run tests
    results = {}
    
    # Test 1: Full Ingest (slowest - optional) - Run FIRST to avoid locking issues
    if args.run_full_ingest:
        results['full_ingest'] = test_cortex_ingest_full(ops)
    else:
        print_info("\nSkipping full ingest test (use --run-full-ingest to run)")

    # Test 2: Get Stats (fastest)
    results['stats'] = test_cortex_get_stats(ops)
    
    if args.stats_only:
        print_info("Skipping Query and Incremental tests due to --stats-only")
    else:
        # Test 3: Query (fast)
        results['query'] = test_cortex_query(ops)
        
        # Test 4: Incremental Ingest (medium)
        results['incremental'] = test_cortex_ingest_incremental(ops)

    
    # Print summary
    print(f"\n{Colors.BOLD}{'='*60}")
    print("Test Summary")
    print(f"{'='*60}{Colors.RESET}\n")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {test_name:20s} {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed_tests}/{total_tests} tests passed{Colors.RESET}\n")
    
    # Exit code
    sys.exit(0 if passed_tests == total_tests else 1)


if __name__ == "__main__":
    main()

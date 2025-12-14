#!/usr/bin/env python3
"""
Test Harness - Systematic Test Execution
Runs all tests in order: Unit → Integration → E2E

Usage:
    python3 tests/run_all_tests.py                           # Run all tests for all servers
    python3 tests/run_all_tests.py --server git              # Run all tests for git server only
    python3 tests/run_all_tests.py --layer unit              # Run only unit tests for all servers
    python3 tests/run_all_tests.py --server git --layer unit # Run only git unit tests
    python3 tests/run_all_tests.py --list                    # List available servers
    python3 tests/run_all_tests.py --no-stop                 # Continue to next layer even if failures

Workflow for Fixing Tests:
    1. Test one server at a time:
       python3 tests/run_all_tests.py --server git --layer unit
    
    2. When unit tests pass, test integration:
       python3 tests/run_all_tests.py --server git --layer integration
    
    3. Move to next server:
       python3 tests/run_all_tests.py --server rag_cortex --layer unit
    
    4. When all servers pass individually, run full suite:
       python3 tests/run_all_tests.py

Servers (simplest → most complex):
    config, adr, protocol, chronicle, task, code, git,
    agent_persona, forge_llm, council, orchestrator, rag_cortex

Layers:
    unit        - Fast, isolated, mocked dependencies
    integration - Real local services (ChromaDB, Ollama, Git)
    e2e         - Full MCP protocol lifecycle (all 12 servers running)
"""
import subprocess
import sys
import argparse
from pathlib import Path

class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*80}")
    print(f"{text}")
    print(f"{'='*80}{Colors.RESET}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

# All MCP servers in order from simplest to most complex
ALL_SERVERS = [
    "config",      # Simplest
    "adr",
    "protocol",
    "chronicle",
    "task",
    "code",
    "git",
    "agent_persona",
    "forge_llm",
    "council",
    "orchestrator",
    "rag_cortex"   # Most complex
]

# Servers with slow LLM-calling integration tests (skip by default)
SLOW_SERVERS = ["agent_persona", "council", "orchestrator"]

ALL_LAYERS = ["unit", "integration", "e2e"]

import re

def parse_pytest_summary(output):
    """
    Parse pytest output to extract test counts.
    
    Returns:
        dict with 'passed', 'failed', 'skipped', 'errors', 'total'
    """
    # Match patterns like "5 passed", "2 failed", "1 skipped", "3 errors"
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    
    # Look for the summary line like "===== 5 passed in 0.12s ====="
    # or "===== 2 failed, 5 passed in 1.23s ====="
    summary_pattern = r'=+\s*([\d\w\s,]+)\s+in\s+'
    match = re.search(summary_pattern, output)
    
    if match:
        summary = match.group(1)
        # Parse individual counts
        if m := re.search(r'(\d+)\s+passed', summary):
            passed = int(m.group(1))
        if m := re.search(r'(\d+)\s+failed', summary):
            failed = int(m.group(1))
        if m := re.search(r'(\d+)\s+skipped', summary):
            skipped = int(m.group(1))
        if m := re.search(r'(\d+)\s+error', summary):
            errors = int(m.group(1))
    
    return {
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'errors': errors,
        'total': passed + failed + skipped + errors
    }

def run_tests(test_path, description, fail_fast=False):
    """
    Run pytest on a specific path.
    
    Args:
        test_path: Path to test directory
        description: Human-readable description
        fail_fast: Stop on first failure
        
    Returns:
        dict with 'success', 'passed', 'failed', 'skipped', 'total'
    """
    print(f"\n{Colors.BOLD}Running: {description}{Colors.RESET}")
    print(f"Path: {test_path}")
    
    cmd = ["pytest", test_path, "-v", "--tb=short"]
    if fail_fast:
        cmd.append("-x")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Parse test counts
    counts = parse_pytest_summary(output)
    
    if result.returncode == 0:
        print_success(f"{description} - {counts['passed']} passed, {counts['skipped']} skipped")
        return {'success': True, **counts}
    else:
        print_error(f"{description} - {counts['failed']} failed, {counts['passed']} passed")
        # Show last 2000 chars of output for failures
        print(output[-2000:])
        return {'success': False, **counts}

def run_layer(project_root, servers, layer_name, layer_display, include_slow=False):
    """Run tests for a specific layer across specified servers."""
    print_header(f"RUNNING: {layer_display}")
    
    if layer_name == "integration":
        print_warning("Note: These tests require ChromaDB (port 8000) and Ollama (port 11434)")
        if not include_slow:
            print_warning("Skipping slow LLM servers (use --slow to include)")
    elif layer_name == "e2e":
        print_warning("Note: These tests require all 12 MCP servers running")
        print_warning("Most E2E tests currently skip (pending MCP client integration)")
    
    suites_passed = 0
    suites_failed = 0
    total_tests_passed = 0
    total_tests_failed = 0
    total_tests_skipped = 0
    skipped_slow = []
    results = {}
    
    for server in servers:
        # Skip slow servers for integration tests unless --slow is passed
        if layer_name == "integration" and server in SLOW_SERVERS and not include_slow:
            skipped_slow.append(server)
            continue
            
        test_path = f"tests/mcp_servers/{server}/{layer_name}/"
        if (project_root / test_path).exists():
            # Check if directory has any test files
            test_files = list((project_root / test_path).glob("test_*.py"))
            if test_files:
                result = run_tests(test_path, f"{server.upper()} {layer_name.title()} Tests")
                results[server] = result
                
                total_tests_passed += result['passed']
                total_tests_failed += result['failed']
                total_tests_skipped += result['skipped']
                
                if result['success']:
                    suites_passed += 1
                else:
                    suites_failed += 1
            else:
                print_warning(f"{server} - No {layer_name} test files found")
        else:
            print_warning(f"{server} - No {layer_name}/ directory")
    
    # Show skipped slow servers
    if skipped_slow:
        print_warning(f"Skipped slow servers: {', '.join(skipped_slow)}")
    
    # Detailed summary
    print(f"\n{Colors.BOLD}{layer_name.title()} Tests Summary:{Colors.RESET}")
    print(f"  Server Suites: {suites_passed}/{suites_passed + suites_failed} passed")
    print(f"  Total Tests:   {total_tests_passed} passed, {total_tests_failed} failed, {total_tests_skipped} skipped")
    
    # Per-server breakdown
    print(f"\n  {Colors.BOLD}Per-Server Breakdown:{Colors.RESET}")
    for server, result in results.items():
        status = Colors.GREEN + "✓" if result['success'] else Colors.RED + "✗"
        print(f"    {status} {server:15} {result['passed']:3} passed, {result['failed']:2} failed, {result['skipped']:2} skipped{Colors.RESET}")
    
    return suites_passed, suites_failed, results, {
        'passed': total_tests_passed,
        'failed': total_tests_failed,
        'skipped': total_tests_skipped
    }

def main():
    """Run tests with command line options."""
    parser = argparse.ArgumentParser(
        description="Project Sanctuary Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tests/run_all_tests.py                        # Run all tests
  python3 tests/run_all_tests.py --server git           # Run git server tests only
  python3 tests/run_all_tests.py --layer unit           # Run only unit tests
  python3 tests/run_all_tests.py --server git --layer unit  # Run git unit tests
  python3 tests/run_all_tests.py --list                 # List available servers
        """
    )
    parser.add_argument(
        "--server", "-s",
        type=str,
        choices=ALL_SERVERS,
        help="Run tests for a specific MCP server only"
    )
    parser.add_argument(
        "--layer", "-l",
        type=str,
        choices=ALL_LAYERS,
        help="Run only a specific test layer (unit, integration, e2e)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available servers and exit"
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Don't stop at layer boundaries on failure (continue to next layer)"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow LLM-calling tests (agent_persona, council, orchestrator integration)"
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        print("Available MCP Servers (simplest → most complex):")
        for i, server in enumerate(ALL_SERVERS, 1):
            slow_marker = " [SLOW]" if server in SLOW_SERVERS else ""
            print(f"  {i:2}. {server}{slow_marker}")
        print("\nAvailable Test Layers:")
        for layer in ALL_LAYERS:
            print(f"  - {layer}")
        print("\nNote: Servers marked [SLOW] have LLM-calling integration tests.")
        print("      Use --slow to include them.")
        return 0
    
    project_root = Path(__file__).parent.parent
    
    # Determine which servers to test
    servers = [args.server] if args.server else ALL_SERVERS
    
    # Determine which layers to test
    layers = [args.layer] if args.layer else ALL_LAYERS
    
    print_header("🧪 PROJECT SANCTUARY TEST HARNESS")
    print(f"Project Root: {project_root}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Servers: {', '.join(servers)}")
    print(f"Layers: {', '.join(layers)}")
    
    total_suites_passed = 0
    total_suites_failed = 0
    total_tests_passed = 0
    total_tests_failed = 0
    total_tests_skipped = 0
    all_results = {}
    
    layer_info = {
        "unit": "UNIT TESTS (Layer 1 - Fast, Isolated)",
        "integration": "INTEGRATION TESTS (Layer 2 - Real Dependencies)",
        "e2e": "E2E TESTS (Layer 3 - Full MCP Protocol)"
    }
    
    for layer in layers:
        suites_passed, suites_failed, results, test_counts = run_layer(
            project_root, 
            servers, 
            layer, 
            layer_info[layer]
        )
        total_suites_passed += suites_passed
        total_suites_failed += suites_failed
        total_tests_passed += test_counts['passed']
        total_tests_failed += test_counts['failed']
        total_tests_skipped += test_counts['skipped']
        all_results[layer] = results
        
        # Stop at layer boundary if failures and not --no-stop
        if suites_failed > 0 and not args.no_stop and layer != layers[-1]:
            print_error(f"\n{layer.title()} tests failed! Stopping before next layer.")
            print_warning("Use --no-stop to continue to next layer despite failures.")
            break
    
    # Final summary
    print_header("FINAL TEST SUMMARY")
    
    total_suites = total_suites_passed + total_suites_failed
    total_tests = total_tests_passed + total_tests_failed + total_tests_skipped
    
    print(f"{Colors.BOLD}Overall Results:{Colors.RESET}")
    print(f"  Server Suites: {total_suites_passed}/{total_suites} passed ({100*total_suites_passed//total_suites if total_suites > 0 else 0}%)")
    print(f"  Total Tests:   {total_tests_passed} passed, {total_tests_failed} failed, {total_tests_skipped} skipped")
    print(f"  Grand Total:   {total_tests} tests")
    
    if total_suites_failed == 0:
        print_success(f"\n🎉 ALL {total_tests_passed} TESTS PASSED! 🎉")
        return 0
    else:
        print_error(f"\n❌ {total_suites_failed} SERVER SUITE(S) FAILED ({total_tests_failed} tests)")
        return 1

if __name__ == "__main__":
    sys.exit(main())


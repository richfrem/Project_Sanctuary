"""
Shared Pytest configuration for all MCP server tests.
Updates the centralized operations_instrumentation.json.
"""
import pytest
import json
import re
from datetime import datetime
from pathlib import Path

# Centralized instrumentation file
INSTRUMENTATION_FILE = Path(__file__).parent / "operations_instrumentation.json"

def load_instrumentation():
    """Load instrumentation data from file."""
    if INSTRUMENTATION_FILE.exists():
        return json.loads(INSTRUMENTATION_FILE.read_text())
    return {"servers": {}, "last_updated": None}

def save_instrumentation(data):
    """Save instrumentation data to file."""
    data["last_updated"] = datetime.now().isoformat()
    INSTRUMENTATION_FILE.write_text(json.dumps(data, indent=2))

def extract_server_from_nodeid(nodeid):
    """Extract server name from test nodeid."""
    # Pattern: tests/mcp_servers/<server>/...
    match = re.search(r'mcp_servers/([^/]+)/', nodeid)
    return match.group(1) if match else None

def extract_operation_from_test(test_name):
    """Extract operation name from test function name."""
    # Pattern: test_<operation> or test_<operation>_xxx
    if test_name.startswith("test_"):
        # Remove test_ prefix and any suffix after the operation name
        op_part = test_name[5:]
        return op_part
    return test_name


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture individual test results and update instrumentation."""
    outcome = yield
    report = outcome.get_result()
    
    # Only process after the test call phase (not setup/teardown)
    if call.when != "call":
        return
    
    # Extract server and operation from test nodeid
    server = extract_server_from_nodeid(item.nodeid)
    if not server:
        return
    
    operation = extract_operation_from_test(item.name)
    
    # Load and update instrumentation
    data = load_instrumentation()
    
    if server not in data["servers"]:
        data["servers"][server] = {"operations": {}}
    
    server_ops = data["servers"][server]["operations"]
    target_key = operation

    # Smart Matching Logic:
    # 1. Check if 'operation' key exists (exact match)
    # 2. Check if '{server}_{operation}' key exists (canonical match)
    # 3. Otherwise, create new entry with 'operation' name
    if operation not in server_ops:
        canonical_name = f"{server}_{operation}"
        if canonical_name in server_ops:
            target_key = canonical_name
    
    if target_key not in server_ops:
        server_ops[target_key] = {}
    
    # Update the operation entry
    server_ops[target_key].update({
        "last_tested": datetime.now().isoformat(),
        "result": report.outcome.upper(),  # PASSED, FAILED, SKIPPED
        "duration": round(report.duration, 3)
    })
    
    save_instrumentation(data)

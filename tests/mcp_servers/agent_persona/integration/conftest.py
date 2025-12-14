"""
Pytest configuration for Agent Persona integration tests.
Includes automatic test result logging.
"""
import pytest
import json
from datetime import datetime
from pathlib import Path

# Test history file location
HISTORY_FILE = Path(__file__).parent / "test_history.json"

def load_history():
    """Load test history from file."""
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return {"runs": []}

def save_history(history):
    """Save test history to file."""
    HISTORY_FILE.write_text(json.dumps(history, indent=2))

@pytest.fixture(scope="session", autouse=True)
def log_test_run(request):
    """Log test run results after all tests complete."""
    yield  # Run tests first
    
    # After all tests complete, log results
    history = load_history()
    
    # Get test results from terminal reporter
    terminalreporter = request.config.pluginmanager.getplugin("terminalreporter")
    
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))
    
    # Create run entry
    run_entry = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": passed + failed + skipped,
        "result": "PASS" if failed == 0 else "FAIL",
        "tests": {}
    }
    
    # Log individual test results
    for status in ["passed", "failed", "skipped"]:
        for report in terminalreporter.stats.get(status, []):
            test_name = report.nodeid.split("::")[-1]
            run_entry["tests"][test_name] = {
                "status": status.upper(),
                "duration": round(report.duration, 3)
            }
    
    # Add to history (keep last 20 runs)
    history["runs"].insert(0, run_entry)
    history["runs"] = history["runs"][:20]
    history["last_updated"] = datetime.now().isoformat()
    
    save_history(history)
    print(f"\n📝 Test results logged to: {HISTORY_FILE}")

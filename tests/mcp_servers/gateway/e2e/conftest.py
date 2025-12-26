"""
Pytest configuration and fixtures for Gateway E2E tests.

Provides:
- GatewayTestClient fixture for all tests
- Execution logging hooks
- Per-test learning package protection verification
"""
import hashlib
import os
import pytest
import time
from pathlib import Path
from typing import Generator

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parents[4]))

from tests.mcp_servers.gateway.lib.gateway_test_client import GatewayTestClient
from tests.mcp_servers.gateway.e2e.execution_log import (
    ExecutionLogger,
    get_execution_logger,
    reset_execution_logger,
)


# Project root for path conversions
PROJECT_ROOT = Path(__file__).parents[4]


def to_container_path(local_path: Path | str) -> str:
    """
    Convert a local filesystem path to the container path.
    
    Containers mount the project at /app, so:
    /Users/.../Project_Sanctuary/tests/... -> /app/tests/...
    """
    local_path = Path(local_path)
    try:
        relative = local_path.relative_to(PROJECT_ROOT)
        return f"/app/{relative}"
    except ValueError:
        # Path not under project root, return as-is
        return str(local_path)


# Learning package directory to protect
LEARNING_DIR = Path(__file__).parents[4] / ".agent" / "learning"

# Tools that legitimately modify learning package (Protocol 128)
# Pattern matches test method names (use underscores, not hyphens)
LEARNING_MODIFIERS = {
    "learning_debrief",     # Protocol 128 learning scan
    "capture_snapshot",     # Protocol 128 snapshot capture
}


def compute_directory_hash(directory: Path) -> str:
    """Compute a hash of all files in a directory for integrity verification."""
    if not directory.exists():
        return "DIRECTORY_NOT_FOUND"
    
    hasher = hashlib.md5()
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            hasher.update(file_path.name.encode())
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def test_modifies_learning(test_name: str) -> bool:
    """Check if a test involves tools that legitimately modify learning."""
    return any(modifier in test_name for modifier in LEARNING_MODIFIERS)


@pytest.fixture(scope="session")
def learning_package_hash() -> str:
    """Capture hash of learning package before any tests run."""
    return compute_directory_hash(LEARNING_DIR)


@pytest.fixture(scope="module")
def gateway_client() -> GatewayTestClient:
    """Provide a GatewayTestClient instance for E2E tests."""
    return GatewayTestClient()


@pytest.fixture(scope="session")
def execution_logger() -> Generator[ExecutionLogger, None, None]:
    """
    Provide execution logger that persists across the entire test session.
    Saves execution log at the end of the session.
    """
    reset_execution_logger()
    logger = get_execution_logger()
    yield logger
    # Save execution log after all tests complete
    log_path = logger.save()
    print(f"\n📋 Execution log saved to: {log_path}")
    summary = logger.get_summary()
    print(f"📊 Results: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1f}%)")


@pytest.fixture
def logged_call(gateway_client, execution_logger, request, learning_package_hash):
    """
    Factory fixture for making logged Gateway calls.
    
    Features:
    1. Records timestamp before call
    2. Makes the Gateway RPC call
    3. Records output/error and duration
    4. Verifies learning package unchanged (unless test modifies learning)
    
    Returns the result dict.
    """
    test_name = request.node.name
    should_protect = not test_modifies_learning(test_name)
    
    # Capture initial state for per-test verification
    initial_hash = compute_directory_hash(LEARNING_DIR) if should_protect else None
    
    def _call(tool_name: str, args: dict):
        start_time = time.time()
        
        try:
            result = gateway_client.call(tool_name, args)
            duration_ms = (time.time() - start_time) * 1000
            
            execution_logger.log_execution(
                tool_name=tool_name,
                input_args=args,
                output=result if result.get("success") else None,
                error=result.get("error") if not result.get("success") else None,
                duration_ms=duration_ms
            )
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            execution_logger.log_execution(
                tool_name=tool_name,
                input_args=args,
                error=str(e),
                duration_ms=duration_ms
            )
            return {"success": False, "error": str(e)}
    
    yield _call
    
    # Per-test learning verification (unless test legitimately modifies learning)
    if should_protect and initial_hash is not None:
        final_hash = compute_directory_hash(LEARNING_DIR)
        if initial_hash != final_hash:
            pytest.fail(
                f"⚠️  Test '{test_name}' modified .agent/learning/ unexpectedly.\n"
                f"If this is intentional, add the tool name to LEARNING_MODIFIERS in conftest.py"
            )


@pytest.fixture(scope="session", autouse=True)
def session_summary(learning_package_hash):
    """
    Print summary at end of session.
    No longer fails on learning modification - per-test checks handle that.
    """
    yield
    # Just print verification status
    final_hash = compute_directory_hash(LEARNING_DIR)
    status = "✅ unchanged" if final_hash == learning_package_hash else "⚠️  modified (by learning tools)"
    print(f"\n📦 Learning package: {status}")


# Test fixtures directory
TEST_FIXTURES_DIR = Path(__file__).parents[3] / "fixtures" / "test_docs"

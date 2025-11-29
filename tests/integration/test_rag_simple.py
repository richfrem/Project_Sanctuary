"""
Simple RAG integration test - following verify_all.py pattern.
Tests the actual RAG pipeline without complex mocking.
"""
import pytest
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

@pytest.mark.integration
def test_rag_query_via_subprocess():
    """Test RAG query by running main.py as a subprocess (like verify_all.py does)."""
    result = subprocess.run(
        [sys.executable, "mnemonic_cortex/app/main.py", "What is Protocol 101?"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Should complete successfully
    assert result.returncode == 0, f"RAG query failed: {result.stderr}"
    
    # Should have output (either answer or error message)
    assert len(result.stdout) > 0, "No output from RAG query"
    
    print(f"✅ RAG query successful:\n{result.stdout[:500]}")

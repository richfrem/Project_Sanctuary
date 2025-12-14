import pytest
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from mnemonic_cortex.app.services.ingestion_service import IngestionService
from mnemonic_cortex.app.synthesis.generator import SynthesisGenerator
from council_orchestrator.orchestrator.memory.cortex import CortexManager

@pytest.mark.integration
def test_strategic_crucible_loop(tmp_path, llm_service):
    """
    Verify the Strategic Crucible Loop:
    1. Gap Analysis (Simulated)
    2. Research (Mocked Intelligence Forge)
    3. Ingestion (Real Cortex Ingestion)
    4. Adaptation (Real Adaptation Packet Generation)
    5. Synthesis (Real Guardian Wakeup)
    """
    project_root = tmp_path
    
    # Setup directories
    (project_root / "01_PROTOCOLS").mkdir(parents=True)
    (project_root / "mnemonic_cortex" / "chroma_db").mkdir(parents=True)
    (project_root / "mnemonic_cortex" / "adaptors" / "packets").mkdir(parents=True)
    (project_root / "WORK_IN_PROGRESS").mkdir(parents=True)
    
    # Setup .env
    env_file = project_root / ".env"
    env_file.write_text(f"DB_PATH=chroma_db\nCHROMA_CHILD_COLLECTION=test_child\nCHROMA_PARENT_STORE=test_parent")

    # --- Step 1: Gap Analysis (Simulated) ---
    print("\n[1] Gap Analysis: Identified need for 'Protocol 777: The Void'")
    
    # --- Step 2: Research (Mocked) ---
    # Create a dummy research report as if produced by Intelligence Forge
    report_path = project_root / "01_PROTOCOLS" / "Protocol_777_The_Void.md"
    report_content = """
# Protocol 777: The Void

## Context
Research indicates a gap in handling null states.

## Decision
We shall embrace the void.

## Consequences
Null pointer exceptions will be transcended.
    """
    report_path.write_text(report_content)
    print(f"\n[2] Research: Generated report at {report_path}")

    import pytest

    # Moved: This test has been relocated to server-specific integration tests.
    # See: tests/mcp_servers/orchestrator/integration/test_strategic_crucible_loop.py

    pytest.skip("Moved to tests/mcp_servers/orchestrator/integration/test_strategic_crucible_loop.py", allow_module_level=True)
    assert ingest_result["added"] == 1

"""
Strategic Crucible Loop E2E Test (Protocol 056/121)
===================================================

Validates the "Self-Evolving Loop" critical capability:
1. Ingest Research  (Medium Memory)
2. Synthesize       (Thinking/Reasoning - Mocked/Stubbed for now)
3. Adapt            (Slow Memory)
4. Cache Wakeup     (Fast Memory)

This is a CRITICAL System Integrity Test.
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest
from mcp_servers.orchestrator.server import orchestrator_run_strategic_cycle
from mcp_servers.rag_cortex.operations import CortexOperations


class TestStrategicCrucibleLoop(BaseIntegrationTest):
    """
    E2E Validation of the Strategic Crucible Loop (Protocol 056).
    """
    
    def get_required_services(self):
        # We need ChromaDB for Cortex ingestion/wakeup
        return [("localhost", 8000, "ChromaDB")]

    @pytest.fixture
    def test_environment(self, tmp_path):
        """Setup isolated environment for the loop."""
        project_root = tmp_path / "project_root"
        project_root.mkdir()
        
        # Create structure
        (project_root / "01_PROTOCOLS").mkdir()
        (project_root / "00_CHRONICLE").mkdir()
        (project_root / "WORK_IN_PROGRESS").mkdir()
        (project_root / ".env").write_text("CHROMA_HOST=localhost\nCHROMA_PORT=8000\n")
        
        return project_root

    def test_protocol_056_loop_execution(self, test_environment):
        """
        Execute the full Strategic Crucible Loop.
        
        We mock internal CortexOperations ONLY for the specific 'ingest' calls if needed, 
        but ideally we use REAL CortexOperations pointed at a temp collection.
        """
        print(f"\n⚡️ Starting Protocol 056 Validation in {test_environment}...")
        
        # 1. Setup Input: A new Research Report
        report_path = test_environment / "01_PROTOCOLS" / "Protocol_777_Test.md"
        report_path.write_text("""
# Protocol 777: Test Loop

## Context
Integration testing of Protocol 056.

## Decision
The loop shall be verified.
""")
        
        # 2. Patch PROJECT_ROOT in Orchestrator to use our temp env
        # Also patch CortexOperations to force it to use test collections
        with patch("mcp_servers.orchestrator.server.PROJECT_ROOT", str(test_environment)):
            
            # We also need to patch how CortexOperations is instantiated inside the server
            # to ensure we use isolated collections (otherwise we write to real DB).
            # This is tricky because server.py instantiates CortexOperations(PROJECT_ROOT).
            
            # Pattern: Mock the class, but have side_effect return a real instance
            # configured safely.
            
            real_cortex_cls = CortexOperations
            
            def safe_cortex_factory(root, **kwargs):
                # Instantiate real ops
                ops = real_cortex_cls(root, **kwargs)
                # OVERRIDE collections to safety
                ops.child_collection_name = "test_loop_child"
                ops.parent_collection_name = "test_loop_parent"
                # Ensure we clean up later? relying on teardown hooks or ignoring
                return ops

            with patch("mcp_servers.orchestrator.server.CortexOperations", side_effect=safe_cortex_factory):
                
                # 3. EXECUTE THE LOOP
                result = orchestrator_run_strategic_cycle(
                    gap_description="Integration Test Gap",
                    research_report_path=str(report_path),
                    days_to_synthesize=1
                )
                
                print(f"\n📝 Loop Result Output:\n{result}")

                # 4. Global Assertions
                assert "Strategic Crucible Cycle" in result
                assert "Ingesting Report" in result
                assert "Cycle Complete" in result
                
                # 5. Verify Artifacts
                # Ingestion should have happened (we can verify by querying if we want, 
                # or trust the output log from Cortex which says 'Ingestion Complete')
                assert "Ingestion Complete" in result
                assert "Cache Updated" in result

    def teardown_method(self, method):
        # Cleanup Chroma collections "test_loop_child" etc if possible
        # For now we assume they are transient or don't block subsequent runs
        pass

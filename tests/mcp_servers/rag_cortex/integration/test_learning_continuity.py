
import pytest
import time
from pathlib import Path
from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest
from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.lib.env_helper import get_env_variable

class TestLearningContinuity(BaseIntegrationTest):
    """
    Integration tests for Protocol 128 (Learning Continuity).
    Validates that Guardian Wakeup correctly ingests learning debriefs.
    """
    
    def get_required_services(self):
        chroma_host = get_env_variable("CHROMA_HOST", required=False) or "127.0.0.1"
        chroma_port = int(get_env_variable("CHROMA_PORT", required=False) or "8110")
        return [(chroma_host, chroma_port, "ChromaDB")]

    @pytest.fixture
    def cortex_ops(self, tmp_path):
        project_root = tmp_path / "project_root"
        project_root.mkdir()
        
        # Setup env
        chroma_host = get_env_variable("CHROMA_HOST", required=False) or "127.0.0.1"
        chroma_port = get_env_variable("CHROMA_PORT", required=False) or "8110"
        (project_root / ".env").write_text(f"CHROMA_HOST={chroma_host}\nCHROMA_PORT={chroma_port}\n")
        
        # Setup Dirs for Protocol 128
        (project_root / ".agent").mkdir()
        (project_root / ".agent" / "learning").mkdir()
        (project_root / ".agent" / "data").mkdir() # For integrity checks
        (project_root / "WORK_IN_PROGRESS").mkdir() # For digest output
        
        # Setup Scripts for Tool Execution
        import shutil
        real_scripts_dir = Path.cwd() / "scripts"
        target_scripts_dir = project_root / "scripts"
        target_scripts_dir.mkdir(exist_ok=True)
        
        if (real_scripts_dir / "capture_code_snapshot.py").exists():
             shutil.copy(real_scripts_dir / "capture_code_snapshot.py", target_scripts_dir / "capture_code_snapshot.py")
        
        # Symlink node_modules for JS dependencies
        real_node_modules = Path.cwd() / "node_modules"
        target_node_modules = project_root / "node_modules"
        if real_node_modules.exists() and not target_node_modules.exists():
            target_node_modules.symlink_to(real_node_modules)
        
        ops = CortexOperations(str(project_root))
        return ops

    def test_guardian_wakeup_standby(self, cortex_ops):
        """
        Verify Guardian Wakeup in 'Standby' state (no debrief file).
        """
        response = cortex_ops.guardian_wakeup()
        assert response.status == "success"
        
        digest_path = Path(response.digest_path)
        assert digest_path.exists()
        content = digest_path.read_text()
        
        # Debugging output
        print(f"\nDistilled Content (Standby):\n{content}\n")
        
        # Should NOT include Section IV. Learning Continuity
        assert "IV. Learning Continuity" not in content
        # Should report Standby in Poka-Yoke
        # Use substring match to be safe against markdown formatting
        assert "Learning Stream" in content
        assert "Standby" in content

    def test_learning_debrief_package(self, cortex_ops):
        """
        Verify the high-fidelity 'Liquid Information' package generation.
        """
        package = cortex_ops.learning_debrief(hours=1)
        
        # Debugging output
        print(f"\nLiquid Information Package:\n{package[:500]}...\n")
        
        assert "[DRAFT] Learning Package Snapshot" in package
        assert "Tactical Evidence" in package
        assert "Architecture Alignment" in package

    def test_guardian_wakeup_active(self, cortex_ops):
        """
        Verify Guardian Wakeup in 'Active' state (debrief file exists).
        """
        # Create a manual debrief file for testing ingestion (since the tool now returns only a string)
        debrief_dir = cortex_ops.project_root / ".agent" / "learning"
        debrief_dir.mkdir(parents=True, exist_ok=True)
        debrief_file = debrief_dir / "learning_debrief.md"
        
        debrief_content = "# Test Debrief\n- Insight: Validated Protocol 128 Integration"
        debrief_file.write_text(debrief_content)
        
        response = cortex_ops.guardian_wakeup()
        assert response.status == "success"
        
        digest_path = Path(response.digest_path)
        content = digest_path.read_text()
        
        # Debugging output
        print(f"\nDistilled Content (Active):\n{content}\n")
        
        # Should INCLUDE Section IV. Learning Continuity
        assert "IV. Learning Continuity" in content
        assert "Protocol 128 Active" in content
        assert "Insight: Validated Protocol 128 Integration" in content
        # Should report Active in Poka-Yoke
        assert "Learning Stream" in content
        assert "**Learning Stream:** Active" in content or "Learning Stream: Active" in content.replace("*", "")

    def test_capture_snapshot_tool(self, cortex_ops):
        """
        Verify the 'cortex_capture_snapshot' tool functionality.
        """
        # Create dummy files for snapshot
        test_file = cortex_ops.project_root / "test_file.py"
        test_file.write_text("print('hello world')")
        
        # Initialize Git repo to satisfy zero-trust checks
        import subprocess
        subprocess.run(["git", "init"], cwd=str(cortex_ops.project_root), check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(cortex_ops.project_root), check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(cortex_ops.project_root), check=True)
        subprocess.run(["git", "add", "."], cwd=str(cortex_ops.project_root), check=True)
        # Commit to have a HEAD
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(cortex_ops.project_root), check=True)
        
        # Modify file to create a diff
        test_file.write_text("print('hello modified world')")
        
        # Call the tool with 'audit' type
        response = cortex_ops.capture_snapshot(
            manifest_files=["test_file.py"],
            snapshot_type="audit",
            strategic_context="Testing Phase"
        )
        
        assert response.status == "success"
        assert response.snapshot_type == "audit"
        assert response.manifest_verified is True
        assert "Verified: 1 files" in response.git_diff_context
        
        # Verify Audit Packet content
        snapshot_path = Path(response.snapshot_path)
        assert snapshot_path.exists()
        content = snapshot_path.read_text()
        
        assert "# Red Team Audit Packet" in content
        assert "Testing Phase" in content
        assert "print('hello modified world')" in content
        
        # Test 'seal' type
        response_seal = cortex_ops.capture_snapshot(
            manifest_files=["test_file.py"],
            snapshot_type="seal"
        )
        
        assert response_seal.status == "success"
        assert response_seal.snapshot_type == "seal"
        assert "learning_package_snapshot.md" in response_seal.snapshot_path

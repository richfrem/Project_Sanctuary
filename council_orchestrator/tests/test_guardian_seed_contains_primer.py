# council_orchestrator/tests/test_guardian_seed_contains_primer.py
# Tests that Guardian awakening seeds contain the wakeup primer

import pytest
import subprocess
import os
import tempfile
import shutil
from pathlib import Path


class TestGuardianSeedContainsPrimer:
    """Test that Guardian seeds contain the wakeup primer after snapshot generation."""

    def test_guardian_seed_includes_wakeup_primer(self):
        """Test that running the snapshot script includes wakeup primer in Guardian seed."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy necessary files to temp directory for isolated testing
            project_root = Path("/Users/richardfremmerlid/Projects/Project_Sanctuary")
            temp_project = Path(temp_dir)

            # Copy package.json for dependencies
            shutil.copy(project_root / "package.json", temp_project / "package.json")
            
            # Install dependencies
            install_result = subprocess.run(["npm", "install"], cwd=temp_project, capture_output=True, text=True, timeout=60)
            if install_result.returncode != 0:
                pytest.fail(f"Failed to install dependencies: {install_result.stderr}")
            
            # Copy the snapshot script
            shutil.copy(project_root / "capture_code_snapshot.js", temp_project / "capture_code_snapshot.js")

            # Create minimal directory structure
            (temp_project / "dataset_package").mkdir()
            (temp_project / "council_orchestrator").mkdir()

            # Change to temp directory and run the script
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_project)

                # Run the snapshot script for council_orchestrator
                result = subprocess.run([
                    "node", "capture_code_snapshot.js", "council_orchestrator"
                ], capture_output=True, text=True, timeout=30)

                # Check that the script ran successfully
                assert result.returncode == 0, f"Script failed: {result.stderr}"

                # Check that Guardian seed was created
                guardian_seed = temp_project / "dataset_package" / "core_essence_guardian_awakening_seed.txt"
                assert guardian_seed.exists(), "Guardian seed file was not created"

                # Read the seed content
                seed_content = guardian_seed.read_text()

                # Verify it contains the wakeup primer
                assert "GUARDIAN WAKEUP PRIMER" in seed_content
                assert "cache_wakeup" in seed_content
                assert "Protocol 114" in seed_content
                assert '"task_type": "cache_wakeup"' in seed_content
                assert "WORK_IN_PROGRESS/guardian_boot_digest.md" in seed_content

            finally:
                os.chdir(original_cwd)

    def test_snapshot_script_has_wakeup_primer_definition(self):
        """Test that the snapshot script contains the guardianWakeupPrimer definition."""
        script_path = Path("/Users/richardfremmerlid/Projects/Project_Sanctuary/capture_code_snapshot.js")

        script_content = script_path.read_text()

        # Verify the primer definition exists
        assert "const GUARDIAN_WAKEUP_PRIMER" in script_content
        assert "GUARDIAN WAKEUP PRIMER" in script_content
        assert "Protocol 114" in script_content
        assert "cache_wakeup" in script_content

    def test_guardian_mandates_include_wakeup_primer(self):
        """Test that Guardian-specific mandates include the wakeup primer."""
        script_path = Path("/Users/richardfremmerlid/Projects/Project_Sanctuary/capture_code_snapshot.js")

        script_content = script_path.read_text()

        # Find the Guardian mandate addition
        guardian_section = None
        lines = script_content.split('\n')
        in_guardian_block = False
        for i, line in enumerate(lines):
            if "if (role.toLowerCase() === 'guardian')" in line:
                in_guardian_block = True
                guardian_section = []
            elif in_guardian_block and line.strip().startswith('}'):
                break
            elif in_guardian_block:
                guardian_section.append(line)

        assert guardian_section is not None, "Guardian mandate block not found"
        guardian_code = '\n'.join(guardian_section)

        # Verify wakeup primer is included
        assert "GUARDIAN_WAKEUP_PRIMER" in guardian_code
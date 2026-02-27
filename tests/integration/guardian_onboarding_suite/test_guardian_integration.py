import pytest
import sys
import tempfile
import json
from pathlib import Path

# Adjust imports for the unified plugins testing environment
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "plugins" / "guardian-onboarding" / "skills" / "forge-soul-exporter" / "scripts"))

from forge_soul import git_preflight, find_sealed_notes, capture_snapshot, verify_snapshot, format_record

def seed_edge_cases(vault_root: Path):
    """Seed synthetic vault with highly malformed margin cases to prove parser resilience."""
    # 1. Malformed YAML (unquoted colons inside strings)
    malformed = "---\ntitle: The Return: Part 2\ntags: [broken, yaml]\nstatus: sealed\n---\n# Malformed Frontmatter\nThis should degrade gracefully.\n"
    (vault_root / "malformed_yaml.md").write_text(malformed)

@pytest.fixture
def synthetic_vault():
    """Creates a temporary isolated vault."""
    with tempfile.TemporaryDirectory(prefix="sanctuary_vault_") as temp_dir:
        vault_root = Path(temp_dir)
        (vault_root / ".obsidian").mkdir()
        (vault_root / "Home.md").write_text("---\ntags: [core]\nstatus: sealed\n---\n# Home\n")
        yield vault_root

class TestGuardianIntegrationSuite:
    def test_forge_soul_export_pipeline(self, synthetic_vault):
        """End-to-End test of the Forge Soul export pipeline."""
        seed_edge_cases(synthetic_vault)
        
        # 1. Git Preflight
        preflight = git_preflight(synthetic_vault)
        assert preflight["clean"] is True
        
        # 2. Find sealed notes
        sealed = find_sealed_notes(synthetic_vault)
        # malformed_yaml.md is skipped because its frontmatter can't be parsed reliably
        assert len(sealed) == 1
        assert "Home.md" in sealed[0]["filepath"]
        
        # 3. Snapshot Isolation
        snapshot = capture_snapshot(sealed)
        clean, changed = verify_snapshot(snapshot)
        assert clean is True
        
        # 4. JSONL Formatting
        records = [format_record(note, "Test-Repo-1.0") for note in sealed]
        assert len(records) == len(sealed)
        assert records[0]["model_version"] == "Test-Repo-1.0"
        assert "timestamp" in records[0]

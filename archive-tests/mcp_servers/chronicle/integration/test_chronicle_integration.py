import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.chronicle.operations import ChronicleOperations
import os
import shutil

@pytest.mark.integration
class TestChronicleIntegration:
    
    @pytest.fixture
    def temp_chronicle_dir(self, tmp_path):
        """Create a temporary chronicle directory."""
        chronicle_dir = tmp_path / "00_CHRONICLE"
        chronicle_dir.mkdir()
        return chronicle_dir

    def test_chronicle_write_flow(self, temp_chronicle_dir):
        """Test writing to Chronicle from an external source (simulated)."""
        
        # Initialize operations with temp dir
        ops = ChronicleOperations(str(temp_chronicle_dir))
        
        # 1. Create a new entry
        result = ops.create_entry(
            title="Integration Test Entry",
            content="This is a test entry from the integration suite.",
            author="TestRunner",
            classification="internal"
        )
        
        assert result["status"] == "draft"
        assert "001_integration_test_entry.md" in result["file_path"]
        
        # 2. Verify file exists
        entry_path = temp_chronicle_dir / "001_integration_test_entry.md"
        assert entry_path.exists()
        
        content = entry_path.read_text()
        assert "**Title:** Integration Test Entry" in content
        assert "**Author:** TestRunner" in content
        
        # 3. Search for the entry
        search_results = ops.search_entries("Integration Test")
        assert len(search_results) >= 1
        assert search_results[0]["title"] == "Integration Test Entry"

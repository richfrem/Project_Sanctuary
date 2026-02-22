"""
Unit tests for Protocol Operations (Business Logic).
Decoupled from Pydantic Models.
"""
import pytest
import tempfile
import shutil
import os
from unittest.mock import MagicMock, patch
from mcp_servers.protocol.operations import ProtocolOperations

class TestProtocolOperations:
    @pytest.fixture
    def setup_ops(self):
        self.test_dir = tempfile.mkdtemp()
        self.ops = ProtocolOperations(base_dir=self.test_dir)
        # Mock validator to isolate operations logic
        # We patch the instance on self.ops
        self.ops.validator = MagicMock()
        
        yield self.ops
        shutil.rmtree(self.test_dir)

    def test_create_protocol(self, setup_ops):
        """Test creating a protocol file."""
        # Setup validator to pass
        setup_ops.validator.validate_required_fields.return_value = True
        setup_ops.validator.validate_protocol_number.return_value = True

        res = setup_ops.create_protocol(
            number=1,
            title="Test Protocol",
            status="CANONICAL",
            classification="public",
            version="1.0",
            authority="me",
            content="This is the way."
        )

        assert res["protocol_number"] == 1
        assert res["status"] == "CANONICAL"
        assert os.path.exists(res["file_path"])
        
        with open(res["file_path"]) as f:
            content = f.read()
            assert "# Protocol 1: Test Protocol" in content
            assert "**Status:** CANONICAL" in content
            assert "This is the way." in content

    def test_get_protocol(self, setup_ops):
        """Test retrieving a protocol."""
        # Create a dummy file manually to bypass create logic
        fname = "002_test.md"
        fpath = os.path.join(setup_ops.base_dir, fname)
        with open(fpath, "w") as f:
            f.write("# Protocol 2: Read Me\n**Status:** PROPOSED\n---\nContent here.")
        
        proto = setup_ops.get_protocol(2)
        
        assert proto["number"] == 2
        assert proto["title"] == "Read Me"
        assert proto["status"] == "PROPOSED"
        assert proto["content"] == "Content here."

    def test_list_protocols_filter(self, setup_ops):
        """Test listing with status filter."""
        # Create two files
        p1 = os.path.join(setup_ops.base_dir, "001_p1.md")
        with open(p1, "w") as f:
            f.write("# Protocol 1: A\n**Status:** CANONICAL\n---\n.")

        p2 = os.path.join(setup_ops.base_dir, "002_p2.md")
        with open(p2, "w") as f:
            f.write("# Protocol 2: B\n**Status:** PROPOSED\n---\n.")

        active_protos = setup_ops.list_protocols(status="CANONICAL")
        assert len(active_protos) == 1
        assert active_protos[0]["number"] == 1

    def test_update_protocol(self, setup_ops):
        """Test updating fields."""
        # Create initial
        setup_ops.validator.validate_required_fields.return_value = True
        setup_ops.validator.validate_protocol_number.return_value = True
        setup_ops.create_protocol(3, "Old", "PROPOSED", "pub", "1.0", "me", "content")
        
        # Update
        res = setup_ops.update_protocol(3, {"status": "CANONICAL", "title": "New"}, "Approval")
        
        assert res["protocol_number"] == 3
        
        # Verify file change
        updated = setup_ops.get_protocol(3)
        assert updated["title"] == "New"
        assert updated["status"] == "CANONICAL"
        assert updated["content"] == "content" # Unchanged matching create arg

    def test_search_protocols(self, setup_ops):
        """Test full text search."""
        p1 = os.path.join(setup_ops.base_dir, "004_search.md")
        with open(p1, "w") as f:
            f.write("# Protocol 4: Search\n---\nFind the hidden keyword here.")
            
        results = setup_ops.search_protocols("hidden keyword")
        assert len(results) == 1
        assert results[0]["number"] == 4

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

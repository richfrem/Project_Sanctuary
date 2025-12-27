"""
Unit tests for Chronicle Operations (Business Logic).
Decoupled from Pydantic Models and FastMCP Interface.
"""
import unittest
import tempfile
import shutil
import os
from datetime import date
from mcp_servers.chronicle.operations import ChronicleOperations

class TestChronicleOperations(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.ops = ChronicleOperations(self.test_dir)
    
    def tearDown(self):
        # Clean up
        shutil.rmtree(self.test_dir)
    
    def test_create_entry(self):
        """Test creating a valid entry."""
        result = self.ops.create_entry(
            title="First Entry",
            content="This is the content.",
            author="User",
            status="draft",
            classification="internal"
        )
        
        # Verify result dict
        self.assertEqual(result["entry_number"], 1)
        self.assertTrue(os.path.exists(result["file_path"]))
        
        # Verify file content
        with open(result["file_path"], "r") as f:
            content = f.read()
            self.assertIn("**Title:** First Entry", content)
            self.assertIn("This is the content.", content)

    def test_update_entry(self):
        """Test updating an entry."""
        create_res = self.ops.create_entry(
            title="Update Me",
            content="Original content",
            author="User"
        )
        entry_num = create_res["entry_number"]
        
        update_res = self.ops.update_entry(
            entry_number=entry_num,
            updates={"content": "Updated content", "status": "active"},
            reason="Refinement"
        )
        
        self.assertIn("content", update_res["updated_fields"])
        
        # Verify content changed
        entry = self.ops.get_entry(entry_num)
        self.assertEqual(entry["content"], "Updated content")
        self.assertEqual(entry["status"], "active")

    def test_get_entry(self):
        """Test retrieving parsed entry."""
        self.ops.create_entry(title="Getter Test", content="Body", author="Me")
        
        entry = self.ops.get_entry(1)
        self.assertEqual(entry["number"], 1)
        self.assertEqual(entry["title"], "Getter Test")
        self.assertEqual(entry["content"], "Body")

    def test_list_entries(self):
        """Test listing entries."""
        self.ops.create_entry(title="A", content="A", author="A")
        self.ops.create_entry(title="B", content="B", author="B")
        
        entries = self.ops.list_entries()
        self.assertEqual(len(entries), 2)
        # Should be reverse sorted (newest first)
        self.assertEqual(entries[0]["title"], "B")
        self.assertEqual(entries[1]["title"], "A")

    def test_search_entries(self):
        """Test searching content."""
        self.ops.create_entry(title="Search Hit", content="The magic key", author="A")
        self.ops.create_entry(title="Search Miss", content="Nothing here", author="A")
        
        results = self.ops.search_entries("magic key")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Search Hit")

if __name__ == "__main__":
    unittest.main()

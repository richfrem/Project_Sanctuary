"""
Unit tests for Chronicle operations
"""
import unittest
import tempfile
import shutil
import os
from datetime import date
from mcp_servers.document.chronicle.operations import ChronicleOperations


class TestChronicleOperations(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ops = ChronicleOperations(self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_create_entry(self):
        """Test creating a new entry."""
        result = self.ops.create_entry(
            title="Test Entry",
            content="Test content",
            author="Tester",
            status="draft",
            classification="internal"
        )
        
        self.assertEqual(result['entry_number'], 1)
        self.assertTrue(os.path.exists(result['file_path']))
        
        # Verify content
        with open(result['file_path'], 'r') as f:
            content = f.read()
            self.assertIn("# Living Chronicle - Entry 1", content)
            self.assertIn("**Title:** Test Entry", content)
            self.assertIn("**Status:** draft", content)
    
    def test_get_entry(self):
        """Test retrieving an entry."""
        created = self.ops.create_entry("Test", "Content", "Author")
        
        entry = self.ops.get_entry(created['entry_number'])
        self.assertEqual(entry['number'], 1)
        self.assertEqual(entry['title'], "Test")
        self.assertEqual(entry['author'], "Author")
        
    def test_list_entries(self):
        """Test listing entries."""
        self.ops.create_entry("Entry 1", "C1", "A1")
        self.ops.create_entry("Entry 2", "C2", "A2")
        
        entries = self.ops.list_entries()
        self.assertEqual(len(entries), 2)
        # Should be reverse sorted (newest first)
        self.assertEqual(entries[0]['number'], 2)
        
    def test_search_entries(self):
        """Test searching entries."""
        self.ops.create_entry("Alpha", "Contains keyword", "A1")
        self.ops.create_entry("Beta", "Nothing here", "A2")
        
        results = self.ops.search_entries("keyword")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], "Alpha")


if __name__ == "__main__":
    unittest.main()

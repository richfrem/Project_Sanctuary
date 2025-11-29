"""
Unit tests for ADR operations
"""
import unittest
import tempfile
import shutil
import os
from mcp_servers.document.adr.operations import ADROperations


class TestADROperations(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.ops = ADROperations(self.test_dir)
    
    def tearDown(self):
        # Clean up
        shutil.rmtree(self.test_dir)
    
    def test_create_adr(self):
        """Test creating a new ADR."""
        result = self.ops.create_adr(
            title="Test Decision",
            context="This is a test context",
            decision="We decided to test",
            consequences="Testing is good"
        )
        
        self.assertEqual(result['adr_number'], 1)
        self.assertTrue(os.path.exists(result['file_path']))
        self.assertEqual(result['status'], "proposed")
    
    def test_create_adr_sequential_numbering(self):
        """Test ADRs are numbered sequentially."""
        result1 = self.ops.create_adr(
            title="First",
            context="Context 1",
            decision="Decision 1",
            consequences="Consequences 1"
        )
        
        result2 = self.ops.create_adr(
            title="Second",
            context="Context 2",
            decision="Decision 2",
            consequences="Consequences 2"
        )
        
        self.assertEqual(result1['adr_number'], 1)
        self.assertEqual(result2['adr_number'], 2)
    
    def test_get_adr(self):
        """Test retrieving an ADR."""
        # Create an ADR
        created = self.ops.create_adr(
            title="Test ADR",
            context="Test context",
            decision="Test decision",
            consequences="Test consequences"
        )
        
        # Retrieve it
        adr = self.ops.get_adr(created['adr_number'])
        
        self.assertEqual(adr['number'], 1)
        self.assertEqual(adr['title'], "Test ADR")
        self.assertEqual(adr['status'], "proposed")
    
    def test_update_adr_status(self):
        """Test updating ADR status."""
        # Create an ADR
        created = self.ops.create_adr(
            title="Test",
            context="Context",
            decision="Decision",
            consequences="Consequences"
        )
        
        # Update status
        result = self.ops.update_adr_status(
            created['adr_number'],
            "accepted",
            "Implemented successfully"
        )
        
        self.assertEqual(result['old_status'], "proposed")
        self.assertEqual(result['new_status'], "accepted")
    
    def test_list_adrs(self):
        """Test listing ADRs."""
        # Create multiple ADRs
        self.ops.create_adr("ADR 1", "C1", "D1", "Cons1")
        self.ops.create_adr("ADR 2", "C2", "D2", "Cons2", status="accepted")
        
        # List all
        all_adrs = self.ops.list_adrs()
        self.assertEqual(len(all_adrs), 2)
        
        # List by status
        accepted = self.ops.list_adrs(status="accepted")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['title'], "ADR 2")
    
    def test_search_adrs(self):
        """Test searching ADRs."""
        # Create ADRs with searchable content
        self.ops.create_adr(
            "FastAPI Decision",
            "We need a web framework",
            "Use FastAPI",
            "Fast and modern"
        )
        self.ops.create_adr(
            "Database Choice",
            "Need a database",
            "Use PostgreSQL",
            "Reliable"
        )
        
        # Search
        results = self.ops.search_adrs("FastAPI")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['number'], 1)


if __name__ == "__main__":
    unittest.main()

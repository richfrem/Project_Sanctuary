"""
Unit tests for Protocol operations
"""
import unittest
import tempfile
import shutil
from mcp_servers.protocol.operations import ProtocolOperations


class TestProtocolOperations(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ops = ProtocolOperations(self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_create_protocol(self):
        """Test creating a new protocol."""
        result = self.ops.create_protocol(
            number=117,
            title="Test Protocol",
            status="CANONICAL",
            classification="Test Framework",
            version="1.0",
            authority="Test Authority",
            content="Test content"
        )
        
        self.assertEqual(result['protocol_number'], 117)
        self.assertEqual(result['status'], "CANONICAL")
        
    def test_get_protocol(self):
        """Test retrieving a protocol."""
        self.ops.create_protocol(
            117, "Test", "CANONICAL", "Framework", "1.0", "Auth", "Content"
        )
        
        protocol = self.ops.get_protocol(117)
        self.assertEqual(protocol['number'], 117)
        self.assertEqual(protocol['title'], "Test")
        
    def test_list_protocols(self):
        """Test listing protocols."""
        self.ops.create_protocol(100, "P1", "CANONICAL", "F1", "1.0", "A1", "C1")
        self.ops.create_protocol(101, "P2", "PROPOSED", "F2", "1.0", "A2", "C2")
        
        all_protocols = self.ops.list_protocols()
        self.assertEqual(len(all_protocols), 2)
        
        canonical = self.ops.list_protocols(status="CANONICAL")
        self.assertEqual(len(canonical), 1)
        
    def test_search_protocols(self):
        """Test searching protocols."""
        self.ops.create_protocol(100, "Alpha", "CANONICAL", "F", "1.0", "A", "Contains keyword")
        self.ops.create_protocol(101, "Beta", "CANONICAL", "F", "1.0", "A", "Nothing here")
        
        results = self.ops.search_protocols("keyword")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], "Alpha")


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Protocol validator
"""
import unittest
import tempfile
import shutil
import os
from mcp_servers.protocol.validator import ProtocolValidator


class TestProtocolValidator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.validator = ProtocolValidator(self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_validate_protocol_number_duplicate(self):
        """Test duplicate protocol number validation."""
        open(os.path.join(self.test_dir, "100_test.md"), 'w').close()
        
        with self.assertRaises(ValueError):
            self.validator.validate_protocol_number(100)
            
    def test_validate_required_fields(self):
        """Test required fields validation."""
        self.validator.validate_required_fields(
            "Title", "Classification", "1.0", "Authority", "Content"
        )
        
        with self.assertRaises(ValueError):
            self.validator.validate_required_fields(
                "", "Classification", "1.0", "Authority", "Content"
            )


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Chronicle validator
"""
import unittest
import tempfile
import shutil
import os
import time
from datetime import datetime, timedelta
from mcp_servers.chronicle.validator import ChronicleValidator


class TestChronicleValidator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.validator = ChronicleValidator(self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_get_next_entry_number(self):
        """Test getting next entry number."""
        self.assertEqual(self.validator.get_next_entry_number(), 1)
        
        # Create some files
        open(os.path.join(self.test_dir, "001_test.md"), 'w').close()
        open(os.path.join(self.test_dir, "002_test.md"), 'w').close()
        
        self.assertEqual(self.validator.get_next_entry_number(), 3)
    
    def test_validate_entry_number_duplicate(self):
        """Test duplicate entry number validation."""
        open(os.path.join(self.test_dir, "001_test.md"), 'w').close()
        
        with self.assertRaises(ValueError):
            self.validator.validate_entry_number(1)
            
    def test_validate_modification_window_new_file(self):
        """Test modification of new file is allowed."""
        file_path = os.path.join(self.test_dir, "001_new.md")
        open(file_path, 'w').close()
        
        # Should not raise
        self.validator.validate_modification_window(file_path)
        
    def test_validate_modification_window_old_file(self):
        """Test modification of old file requires override."""
        file_path = os.path.join(self.test_dir, "001_old.md")
        open(file_path, 'w').close()
        
        # Set mtime to 8 days ago
        old_time = time.time() - (8 * 24 * 3600)
        os.utime(file_path, (old_time, old_time))
        
        # Should raise without override
        with self.assertRaises(ValueError):
            self.validator.validate_modification_window(file_path)
            
        # Should pass with override
        self.validator.validate_modification_window(file_path, override_approval_id="AUTH-123")

    def test_validate_required_fields(self):
        """Test required fields validation."""
        self.validator.validate_required_fields("Title", "Content", "Author")
        
        with self.assertRaises(ValueError):
            self.validator.validate_required_fields("", "Content", "Author")


if __name__ == "__main__":
    unittest.main()

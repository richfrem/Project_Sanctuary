"""
Unit tests for ADR validator
"""
import unittest
import tempfile
import shutil
import os
from mcp_servers.document.adr.validator import ADRValidator
from mcp_servers.document.adr.models import ADRStatus


class TestADRValidator(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.validator = ADRValidator(self.test_dir)
    
    def tearDown(self):
        # Clean up
        shutil.rmtree(self.test_dir)
    
    def test_get_next_adr_number_empty_dir(self):
        """Test getting next ADR number in empty directory."""
        self.assertEqual(self.validator.get_next_adr_number(), 1)
    
    def test_get_next_adr_number_with_existing(self):
        """Test getting next ADR number with existing ADRs."""
        # Create some ADR files
        open(os.path.join(self.test_dir, "001_first.md"), 'w').close()
        open(os.path.join(self.test_dir, "002_second.md"), 'w').close()
        
        self.assertEqual(self.validator.get_next_adr_number(), 3)
    
    def test_validate_adr_number_duplicate(self):
        """Test validation fails for duplicate ADR number."""
        open(os.path.join(self.test_dir, "001_existing.md"), 'w').close()
        
        with self.assertRaises(ValueError) as context:
            self.validator.validate_adr_number(1)
        
        self.assertIn("already exists", str(context.exception))
    
    def test_validate_status_transition_valid(self):
        """Test valid status transitions."""
        # proposed -> accepted
        self.validator.validate_status_transition(
            ADRStatus.PROPOSED, 
            ADRStatus.ACCEPTED
        )
        
        # accepted -> deprecated
        self.validator.validate_status_transition(
            ADRStatus.ACCEPTED,
            ADRStatus.DEPRECATED
        )
    
    def test_validate_status_transition_invalid(self):
        """Test invalid status transitions."""
        with self.assertRaises(ValueError) as context:
            self.validator.validate_status_transition(
                ADRStatus.ACCEPTED,
                ADRStatus.PROPOSED
            )
        
        self.assertIn("Invalid transition", str(context.exception))
    
    def test_validate_supersedes_not_found(self):
        """Test validation fails when superseded ADR doesn't exist."""
        with self.assertRaises(ValueError) as context:
            self.validator.validate_supersedes(999)
        
        self.assertIn("does not exist", str(context.exception))
    
    def test_validate_required_fields(self):
        """Test validation of required fields."""
        # Valid fields
        self.validator.validate_required_fields(
            "Title", "Context", "Decision", "Consequences"
        )
        
        # Empty title
        with self.assertRaises(ValueError):
            self.validator.validate_required_fields(
                "", "Context", "Decision", "Consequences"
            )


if __name__ == "__main__":
    unittest.main()

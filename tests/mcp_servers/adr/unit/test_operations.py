"""
Unit tests for ADR Operations (Business Logic Layer)
Decoupled from Pydantic Models and FastMCP Interface.
"""
import unittest
import tempfile
import shutil
import os
from datetime import datetime
from mcp_servers.adr.operations import ADROperations
from mcp_servers.adr.models import ADRStatus

class TestADROperations(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for ADRs
        self.test_dir = tempfile.mkdtemp()
        self.ops = ADROperations(self.test_dir)
    
    def tearDown(self):
        # Clean up
        shutil.rmtree(self.test_dir)
    
    def test_create_adr_success(self):
        """Test creating a valid ADR using primitive types."""
        result = self.ops.create_adr(
            title="Use Python for Scripts",
            context="We need a scripting language.",
            decision="We will use Python.",
            consequences="Easy to read, popular.",
            author="Test User",
            status="accepted"
        )
        
        # Verify return dict
        self.assertEqual(result["adr_number"], 1)
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(os.path.exists(result["file_path"]))
        
        # Verify file content
        with open(result["file_path"], "r") as f:
            content = f.read()
            self.assertIn("# Use Python for Scripts", content)
            self.assertIn("**Status:** accepted", content)
            self.assertIn("We will use Python.", content)

    def test_create_adr_validation_failure(self):
        """Test that validation logic is invoked."""
        with self.assertRaises(ValueError):
            self.ops.create_adr(
                title="",  # Empty title should fail
                context="Context",
                decision="Decision",
                consequences="Consequences"
            )

    def test_update_adr_status(self):
        """Test status update logic."""
        # 1. Create ADR
        create_res = self.ops.create_adr(
            title="Test Status Update",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            status="proposed"
        )
        adr_num = create_res["adr_number"]
        
        # 2. Update Status
        update_res = self.ops.update_adr_status(
            number=adr_num,
            new_status="accepted",
            reason="Team approved it."
        )
        
        # 3. Verify Return
        self.assertEqual(update_res["old_status"], "proposed")
        self.assertEqual(update_res["new_status"], "accepted")
        
        # 4. Verify File Content
        with open(create_res["file_path"], "r") as f:
            content = f.read()
            self.assertIn("**Status:** accepted", content)
            self.assertIn("**Status Update", content)
            self.assertIn("Team approved it.", content)

    def test_get_adr(self):
        """Test retrieving parsed ADR data."""
        self.ops.create_adr(
            title="Retrieval Test",
            context="Context content",
            decision="Decision content",
            consequences="Consequences content"
        )
        
        data = self.ops.get_adr(1)
        self.assertEqual(data["number"], 1)
        self.assertEqual(data["title"], "Retrieval Test")
        self.assertEqual(data["context"], "Context content")

    def test_list_adrs(self):
        """Test listing functionality."""
        self.ops.create_adr(title="First", context="C", decision="D", consequences="C")
        self.ops.create_adr(title="Second", context="C", decision="D", consequences="C")
        
        adrs = self.ops.list_adrs()
        self.assertEqual(len(adrs), 2)
        self.assertEqual(adrs[0]["title"], "First")
        self.assertEqual(adrs[1]["title"], "Second")

    def test_list_adrs_filter(self):
        """Test listing with status filter."""
        self.ops.create_adr(title="Prop", context="C", decision="D", consequences="C", status="proposed")
        self.ops.create_adr(title="Acc", context="C", decision="D", consequences="C", status="accepted")
        
        proposed = self.ops.list_adrs(status="proposed")
        self.assertEqual(len(proposed), 1)
        self.assertEqual(proposed[0]["title"], "Prop")

    def test_search_adrs(self):
        """Test searching content."""
        self.ops.create_adr(
            title="Search Me",
            context="The quick brown fox",
            decision="jumps over",
            consequences="the lazy dog"
        )
        
        # Match
        results = self.ops.search_adrs("brown fox")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Search Me")
        self.assertIn("The quick brown fox", results[0]["matches"][0])
        
        # No Match
        results = self.ops.search_adrs("flying elephant")
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()

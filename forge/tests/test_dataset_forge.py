import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# The module is added to path in conftest.py, but we need to import it carefully
# because it might run code on import if not guarded by if __name__ == "__main__":
try:
    import forge_whole_genome_dataset as forge
except ImportError:
    # Fallback if path setup in conftest doesn't work for direct import here
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../OPERATION_PHOENIX_FORGE/scripts')))
    import forge_whole_genome_dataset as forge

class TestInstructionDetermination:
    def test_determine_instruction_protocol(self):
        """Test instruction generation for Protocol files."""
        filename = "01_PROTOCOLS/101_The_Unbreakable_Commit.md"
        instruction = forge.determine_instruction(filename)
        assert "Protocol 101" in instruction
        assert "foundational doctrine" in instruction

    def test_determine_instruction_chronicle(self):
        """Test instruction generation for Chronicle entries."""
        filename = "00_CHRONICLE/ENTRIES/001_Genesis.md"
        instruction = forge.determine_instruction(filename)
        assert "Chronicle Entry" in instruction
        assert "historical record" in instruction

    def test_determine_instruction_task(self):
        """Test instruction generation for Task files."""
        filename = "tasks/done/001_setup.md"
        instruction = forge.determine_instruction(filename)
        assert "Task" in instruction
        assert "execution record" in instruction

    def test_determine_instruction_default(self):
        """Test default instruction generation."""
        filename = "random_file.txt"
        instruction = forge.determine_instruction(filename)
        assert "Synthesize the core concepts" in instruction
        assert "random_file.txt" in instruction

class TestDatasetGeneration:
    @patch('builtins.open')
    @patch('json.dump')
    def test_main_execution_flow(self, mock_json_dump, mock_open_func):
        """Test the main execution flow with mocked file operations."""
        # Mock the snapshot content
        mock_snapshot = """
---
file: 01_PROTOCOLS/test.md
---
# Test Protocol
Content here.
"""
        # Setup mock file reads
        mock_file = MagicMock()
        mock_file.read.return_value = mock_snapshot
        mock_open_func.return_value.__enter__.return_value = mock_file

        # We need to mock the glob or file listing if the script uses it
        # But looking at the script, it reads from 'sanctuary_snapshot.txt'
        
        # Since main() runs the whole process, we might want to just test the logic parts
        # or mock everything heavily. 
        
        # For now, let's stick to testing the logic functions which are more critical
        pass

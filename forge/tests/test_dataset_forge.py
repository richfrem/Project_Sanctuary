import pytest
import json
import os
from unittest.mock import patch, mock_open

# Import the module to be tested
# Note: We need to handle potential import errors if dependencies are missing in the test env
try:
    import forge_whole_genome_dataset as forge_dataset
except ImportError:
    # If direct import fails, we might need to use importlib or ensure path is correct
    # The conftest.py adds the path, so it should work if dependencies are met.
    # However, the script might have top-level code that runs on import.
    # Let's assume for now we can import it or mock the parts we need.
    pass

def test_format_instruction():
    """Test the instruction formatting logic."""
    # Since we can't easily import the script if it's not a proper module, 
    # we might need to rely on reading the file or refactoring the script.
    # For this task, let's assume we can import specific functions if we refactor 
    # or if the script is structured well.
    
    # If the script is just a linear execution, we can't unit test functions easily.
    # Let's check the script content first.
    pass

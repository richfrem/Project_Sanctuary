import pytest
import os
import json
import sys
from unittest.mock import MagicMock

# Ensure the scripts directory is in the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../forge/scripts')))

@pytest.fixture
def mock_jsonl_data():
    """Return a list of sample JSONL records."""
    return [
        {"instruction": "Test instruction 1", "input": "Test input 1", "output": "Test output 1"},
        {"instruction": "Test instruction 2", "input": "", "output": "Test output 2"},
    ]

@pytest.fixture
def mock_dataset_file(tmp_path, mock_jsonl_data):
    """Create a temporary JSONL file."""
    file_path = tmp_path / "test_dataset.jsonl"
    with open(file_path, "w") as f:
        for record in mock_jsonl_data:
            f.write(json.dumps(record) + "\n")
    return str(file_path)

@pytest.fixture
def mock_tokenizer():
    """Mock a tokenizer for validation tests."""
    tokenizer = MagicMock()
    # Mock encode to return a list of length proportional to input string length
    tokenizer.encode.side_effect = lambda x: [1] * len(x.split())
    return tokenizer

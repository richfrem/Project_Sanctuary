import pytest
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# Add scripts directory to path to import shim
scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
sys.path.append(str(scripts_dir))

from ingest_code_shim import parse_python_to_markdown, convert_and_save
from mcp_servers.rag_cortex.operations import CortexOperations


@pytest.fixture
def sample_python_file(tmp_path):
    """Create a sample Python file for testing."""
    content = '''
"""
Sample Module Docstring
"""
import os

def hello_world(name: str) -> str:
    """
    Say hello to someone.
    """
    return f"Hello {name}"

class Greeter:
    """
    A class that greets.
    """
    def greet(self):
        print("Greetings")
'''
    file_path = tmp_path / "sample.py"
    file_path.write_text(content.strip(), encoding="utf-8")
    return file_path


def test_shim_conversion_content(sample_python_file):
    """Test that the shim correctly converts code to markdown."""
    markdown = parse_python_to_markdown(str(sample_python_file))
    
    assert "# Code File: sample.py" in markdown
    assert "**Language:** Python" in markdown
    assert "## Module Description" in markdown
    assert "Sample Module Docstring" in markdown
    
    # Check function extraction
    assert "## Function: `hello_world`" in markdown
    assert "**Signature:** `hello_world(name: str) -> str`" in markdown
    assert "Say hello to someone" in markdown
    assert "def hello_world(name: str) -> str:" in markdown
    
    # Check class extraction
    assert "## Class: `Greeter`" in markdown
    assert "A class that greets" in markdown
    assert "**Methods:** `greet`" in markdown


def test_shim_file_saving(sample_python_file):
    """Test that convert_and_save creates the file correctly."""
    output_path = convert_and_save(str(sample_python_file))
    
    assert output_path.endswith(".md")
    assert Path(output_path).exists()
    
    content = Path(output_path).read_text(encoding="utf-8")
    assert "# Code File: sample.py" in content


@pytest.fixture
def mock_cortex_deps():
    """Mock dependencies/internal components of CortexOperations."""
    with patch("langchain_chroma.Chroma"), \
         patch("langchain.storage.LocalFileStore"), \
         patch("langchain.retrievers.ParentDocumentRetriever"), \
         patch("langchain_nomic.NomicEmbeddings"), \
         patch("langchain_community.document_loaders.TextLoader") as mock_text_loader:
         
        yield {
            "text_loader": mock_text_loader
        }

def test_integration_flow(mock_cortex_deps, sample_python_file):
    """
    Test the full flow:
    1. Shim converts .py -> .md
    2. Cortex ingests the .md
    """
    # 1. Convert
    md_file_path = convert_and_save(str(sample_python_file))
    
    # 2. Setup Ingestion mocks
    ops = CortexOperations(str(sample_python_file.parent))
    
    # Mock loader to return what we expect from the md file
    mock_loader = mock_cortex_deps["text_loader"].return_value
    mock_loader.load.return_value = [
        Document(page_content="Mocked markdown content", metadata={"source": md_file_path})
    ]
    
    # 3. Ingest
    result = ops.ingest_incremental(file_paths=[md_file_path])
    
    assert result.status == "success"
    assert result.documents_added == 1
    
    # Clean up
    Path(md_file_path).unlink()

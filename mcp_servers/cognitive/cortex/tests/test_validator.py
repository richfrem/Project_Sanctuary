"""
Unit tests for Cortex MCP validator
"""
import pytest
import tempfile
import os
from pathlib import Path
from mcp_servers.cognitive.cortex.validator import CortexValidator, ValidationError


@pytest.fixture
def temp_project_root():
    """Create a temporary project root for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test directories and files
        protocols_dir = Path(tmpdir) / "01_PROTOCOLS"
        protocols_dir.mkdir()
        
        test_file = protocols_dir / "test.md"
        test_file.write_text("# Test Protocol")
        
        yield tmpdir


def test_validator_init(temp_project_root):
    """Test validator initialization."""
    validator = CortexValidator(temp_project_root)
    assert validator.project_root == Path(temp_project_root)


def test_validate_ingest_full_success(temp_project_root):
    """Test successful validation of ingest_full."""
    validator = CortexValidator(temp_project_root)
    result = validator.validate_ingest_full(
        purge_existing=True,
        source_directories=["01_PROTOCOLS"]
    )
    assert result["purge_existing"] is True
    assert result["source_directories"] == ["01_PROTOCOLS"]


def test_validate_ingest_full_invalid_directory(temp_project_root):
    """Test validation fails for non-existent directory."""
    validator = CortexValidator(temp_project_root)
    with pytest.raises(ValidationError, match="does not exist"):
        validator.validate_ingest_full(
            purge_existing=True,
            source_directories=["NONEXISTENT_DIR"]
        )


def test_validate_query_success(temp_project_root):
    """Test successful validation of query."""
    validator = CortexValidator(temp_project_root)
    result = validator.validate_query(
        query="What is Protocol 101?",
        max_results=5,
        use_cache=False
    )
    assert result["query"] == "What is Protocol 101?"
    assert result["max_results"] == 5
    assert result["use_cache"] is False


def test_validate_query_empty_string(temp_project_root):
    """Test validation fails for empty query."""
    validator = CortexValidator(temp_project_root)
    with pytest.raises(ValidationError, match="cannot be empty"):
        validator.validate_query(query="", max_results=5)


def test_validate_query_whitespace_only(temp_project_root):
    """Test validation fails for whitespace-only query."""
    validator = CortexValidator(temp_project_root)
    with pytest.raises(ValidationError, match="cannot be empty"):
        validator.validate_query(query="   ", max_results=5)


def test_validate_query_too_long(temp_project_root):
    """Test validation fails for query that's too long."""
    validator = CortexValidator(temp_project_root)
    long_query = "x" * 10001
    with pytest.raises(ValidationError, match="too long"):
        validator.validate_query(query=long_query, max_results=5)


def test_validate_query_max_results_too_low(temp_project_root):
    """Test validation fails for max_results < 1."""
    validator = CortexValidator(temp_project_root)
    with pytest.raises(ValidationError, match="must be at least 1"):
        validator.validate_query(query="test", max_results=0)


def test_validate_query_max_results_too_high(temp_project_root):
    """Test validation fails for max_results > 100."""
    validator = CortexValidator(temp_project_root)
    with pytest.raises(ValidationError, match="cannot exceed 100"):
        validator.validate_query(query="test", max_results=101)


def test_validate_ingest_incremental_success(temp_project_root):
    """Test successful validation of ingest_incremental."""
    validator = CortexValidator(temp_project_root)
    test_file = Path(temp_project_root) / "01_PROTOCOLS" / "test.md"
    
    result = validator.validate_ingest_incremental(
        file_paths=[str(test_file)],
        metadata={"author": "test"},
        skip_duplicates=True
    )
    assert len(result["file_paths"]) == 1
    assert result["metadata"]["author"] == "test"
    assert result["skip_duplicates"] is True


def test_validate_ingest_incremental_relative_path(temp_project_root):
    """Test validation converts relative paths to absolute."""
    validator = CortexValidator(temp_project_root)
    
    result = validator.validate_ingest_incremental(
        file_paths=["01_PROTOCOLS/test.md"],
        skip_duplicates=True
    )
    assert len(result["file_paths"]) == 1
    assert os.path.isabs(result["file_paths"][0])


def test_validate_ingest_incremental_empty_list(temp_project_root):
    """Test validation fails for empty file_paths."""
    validator = CortexValidator(temp_project_root)
    with pytest.raises(ValidationError, match="cannot be empty"):
        validator.validate_ingest_incremental(file_paths=[])


def test_validate_ingest_incremental_too_many_files(temp_project_root):
    """Test validation fails for too many files."""
    validator = CortexValidator(temp_project_root)
    file_paths = ["file.md"] * 1001
    with pytest.raises(ValidationError, match="Cannot ingest more than 1000"):
        validator.validate_ingest_incremental(file_paths=file_paths)


def test_validate_ingest_incremental_file_not_exists(temp_project_root):
    """Test validation fails for non-existent file."""
    validator = CortexValidator(temp_project_root)
    with pytest.raises(ValidationError, match="does not exist"):
        validator.validate_ingest_incremental(file_paths=["nonexistent.md"])


def test_validate_ingest_incremental_not_markdown(temp_project_root):
    """Test validation fails for non-markdown file."""
    validator = CortexValidator(temp_project_root)
    test_file = Path(temp_project_root) / "test.txt"
    test_file.write_text("test")
    
    with pytest.raises(ValidationError, match="not a markdown file"):
        validator.validate_ingest_incremental(file_paths=[str(test_file)])


def test_validate_ingest_incremental_invalid_metadata(temp_project_root):
    """Test validation fails for invalid metadata type."""
    validator = CortexValidator(temp_project_root)
    test_file = Path(temp_project_root) / "01_PROTOCOLS" / "test.md"
    
    with pytest.raises(ValidationError, match="must be a dictionary"):
        validator.validate_ingest_incremental(
            file_paths=[str(test_file)],
            metadata="invalid"
        )


def test_validate_stats(temp_project_root):
    """Test validation of stats (no parameters)."""
    validator = CortexValidator(temp_project_root)
    result = validator.validate_stats()
    assert result == {}

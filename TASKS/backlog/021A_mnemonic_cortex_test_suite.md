# Task 021A: Mnemonic Cortex Test Suite

## Metadata
- **Status**: backlog
- **Priority**: high
- **Complexity**: medium
- **Category**: testing
- **Estimated Effort**: 4-6 hours
- **Dependencies**: None
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Parent Task**: 021 (split into 021A, 021B, 021C)

## Context

The `mnemonic_cortex/` module currently has only 2 test files (`test_ingestion.py`, `test_query.py`) providing minimal coverage. This creates risk for regressions and makes refactoring dangerous.

**Strategic Alignment:**
- **Protocol 89**: The Clean Forge - Quality through systematic testing
- **Protocol 101**: The Unbreakable Commit - Verification before commit

## Objective

Create comprehensive unit test suite for the Mnemonic Cortex module achieving 80%+ code coverage.

## Acceptance Criteria

### 1. Core Test Files
- [ ] Create `mnemonic_cortex/tests/test_vector_db_service.py`
  - Test document ingestion
  - Test query functionality
  - Test parent document retrieval
  - Test error handling
- [ ] Create `mnemonic_cortex/tests/test_embedding_service.py`
  - Test embedding generation
  - Test batch processing
  - Test caching behavior
- [ ] Create `mnemonic_cortex/tests/test_cache_manager.py`
  - Test cache hit/miss logic
  - Test TTL expiration
  - Test cache invalidation

### 2. Test Infrastructure
- [ ] Create `mnemonic_cortex/tests/conftest.py` with shared fixtures
- [ ] Add test data fixtures (`tests/fixtures/sample_documents.json`)
- [ ] Create isolated test database setup/teardown
- [ ] Add pytest configuration in `mnemonic_cortex/pytest.ini`

### 3. Coverage & Quality
- [ ] Achieve minimum 80% code coverage for `mnemonic_cortex/core/`
- [ ] All tests pass consistently (no flaky tests)
- [ ] Tests run in under 30 seconds
- [ ] Add coverage report generation

## Technical Approach

```python
# mnemonic_cortex/tests/conftest.py
import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_db_path():
    """Provide temporary database path for isolated testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            "content": "# Protocol 1\n\nTest content for protocol 1.",
            "metadata": {"source": "test_protocol_1.md", "type": "protocol"}
        },
        {
            "content": "# Protocol 2\n\nTest content for protocol 2.",
            "metadata": {"source": "test_protocol_2.md", "type": "protocol"}
        }
    ]

@pytest.fixture
def vector_db_service(temp_db_path, sample_documents):
    """Provide configured vector DB service for testing."""
    from mnemonic_cortex.core.vector_db_service import VectorDBService
    service = VectorDBService(str(temp_db_path))
    service.ingest_documents(sample_documents)
    return service
```

```python
# mnemonic_cortex/tests/test_vector_db_service.py
import pytest

def test_document_ingestion(vector_db_service, sample_documents):
    """Test that documents are ingested correctly."""
    # Query should return ingested documents
    results = vector_db_service.query("Protocol 1", k=1)
    assert len(results) > 0
    assert "Protocol 1" in results[0]["content"]

def test_parent_document_retrieval(vector_db_service):
    """Test that complete parent documents are retrieved."""
    results = vector_db_service.query("test content", k=1)
    
    # Should return full document, not just chunk
    assert len(results[0]["content"]) > 50
    assert results[0]["metadata"]["source"] == "test_protocol_1.md"

def test_query_with_no_results(vector_db_service):
    """Test behavior when query has no matching results."""
    results = vector_db_service.query("nonexistent content xyz123", k=5)
    # Should return empty list or handle gracefully
    assert isinstance(results, list)

def test_error_handling_invalid_k(vector_db_service):
    """Test error handling for invalid k parameter."""
    with pytest.raises(ValueError):
        vector_db_service.query("test", k=0)
    
    with pytest.raises(ValueError):
        vector_db_service.query("test", k=-1)
```

## Success Metrics

- [ ] 80%+ code coverage for `mnemonic_cortex/core/`
- [ ] All tests pass in CI/CD
- [ ] Test execution time < 30 seconds
- [ ] Zero flaky tests

## Related Protocols

- **P89**: The Clean Forge
- **P101**: The Unbreakable Commit
- **P115**: The Tactical Mandate

# Task 021B: Forge Test Suite & CI/CD Pipeline

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

The `forge/scripts/` directory has 6 verification scripts but no unit tests. Additionally, there's no CI/CD pipeline for automated testing across the repository.

## Objective

Create test suite for Forge scripts and establish CI/CD pipeline with GitHub Actions for automated testing.

## Acceptance Criteria

### 1. Forge Test Suite
- [ ] Create `forge/tests/test_dataset_forge.py`
  - Test JSONL dataset generation
  - Test dataset validation
  - Test markdown file processing
- [ ] Create `forge/tests/test_modelfile_generation.py`
  - Test Modelfile creation
  - Test template rendering
  - Test configuration validation
- [ ] Convert existing verification scripts to pytest tests
- [ ] Add mock fixtures for expensive operations (model loading)
- [ ] Achieve 70%+ coverage for `forge/scripts/`

### 2. CI/CD Pipeline
- [ ] Create `.github/workflows/test.yml` for GitHub Actions
- [ ] Configure test matrix (Python 3.11, 3.12; Windows, Linux)
- [ ] Add automated testing on PR creation
- [ ] Add coverage reporting to PR comments
- [ ] Add test status badge to README.md
- [ ] Configure test caching for faster runs

### 3. Test Infrastructure
- [ ] Create `tests/conftest.py` at project root with shared fixtures
- [ ] Create `pytest.ini` at project root with unified configuration
- [ ] Add test execution scripts (`run_tests.sh`, `run_tests.ps1`)
- [ ] Configure coverage reporting (HTML + terminal)

## Technical Approach

```python
# forge/tests/test_dataset_forge.py
import pytest
from pathlib import Path
import json

@pytest.fixture
def sample_markdown_files(tmp_path):
    """Create sample markdown files for testing."""
    protocol_dir = tmp_path / "01_PROTOCOLS"
    protocol_dir.mkdir()
    
    (protocol_dir / "01_test.md").write_text("# Protocol 1\n\nTest content.")
    (protocol_dir / "02_test.md").write_text("# Protocol 2\n\nMore content.")
    
    return tmp_path

def test_dataset_forge_creates_valid_jsonl(sample_markdown_files, tmp_path):
    """Test that dataset forging creates valid JSONL."""
    from forge.scripts.forge_whole_genome_dataset import forge_dataset
    
    output_file = tmp_path / "test_dataset.jsonl"
    forge_dataset(source_dir=sample_markdown_files, output_file=output_file)
    
    assert output_file.exists()
    
    # Validate JSONL format
    with open(output_file) as f:
        for line in f:
            entry = json.loads(line)  # Should not raise
            assert "text" in entry or "messages" in entry

def test_dataset_includes_all_protocols(sample_markdown_files, tmp_path):
    """Test that all protocol files are included."""
    from forge.scripts.forge_whole_genome_dataset import forge_dataset
    
    output_file = tmp_path / "test_dataset.jsonl"
    forge_dataset(sample_markdown_files, output_file)
    
    with open(output_file) as f:
        entries = [json.loads(line) for line in f]
    
    assert len(entries) >= 2
    texts = [e.get("text", "") for e in entries]
    assert any("Protocol 1" in t for t in texts)
    assert any("Protocol 2" in t for t in texts)
```

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ['3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest --cov=. --cov-report=xml --cov-report=term
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-${{ matrix.os }}-py${{ matrix.python-version }}
```

## Success Metrics

- [ ] 70%+ code coverage for `forge/scripts/`
- [ ] CI/CD pipeline running on all PRs
- [ ] Test execution time < 5 minutes
- [ ] All tests passing on Windows and Linux

## Related Protocols

- **P89**: The Clean Forge
- **P101**: The Unbreakable Commit
- **P115**: The Tactical Mandate

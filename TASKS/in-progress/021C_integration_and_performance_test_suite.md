# Task 021C: Integration & Performance Test Suite

## Metadata
- **Status**: backlog
- **Priority**: medium
- **Complexity**: medium
- **Category**: testing
- **Estimated Effort**: 4-6 hours
- **Dependencies**: 021A, 021B
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Parent Task**: 021 (split into 021A, 021B, 021C)

## Context

While unit tests verify individual components, integration tests are needed to verify cross-module workflows and end-to-end functionality. Performance tests establish baselines for critical operations.

## Objective

Create integration test suite for cross-module workflows and performance benchmarks for critical operations.

## Acceptance Criteria

### 1. Integration Tests
- [ ] Create `tests/integration/test_end_to_end_rag_pipeline.py`
  - Test complete RAG flow: ingestion → query → response
  - Test with real protocol documents
  - Verify result quality
- [ ] Create `tests/integration/test_council_orchestrator_with_cortex.py`
  - Test orchestrator querying Cortex
  - Test cache warming workflow
  - Test git operations integration
- [ ] Create `tests/integration/test_secrets_management_integration.py`
  - Test secrets loading across all modules
  - Test fallback chain (Windows → WSL → .env)
  - Test error handling

### 2. Performance Benchmarks
- [ ] Create `tests/benchmarks/test_rag_query_performance.py`
  - Benchmark query latency (p50, p95, p99)
  - Test with varying query complexities
  - Establish baseline metrics
- [ ] Create `tests/benchmarks/test_embedding_generation_speed.py`
  - Benchmark embedding generation
  - Test batch processing performance
  - Compare batch sizes
- [ ] Add pytest-benchmark configuration
- [ ] Create performance baseline report

### 3. Test Organization
- [ ] Mark integration tests with `@pytest.mark.integration`
- [ ] Mark performance tests with `@pytest.mark.benchmark`
- [ ] Configure pytest to skip slow tests by default
- [ ] Add `run_integration_tests.sh` script
- [ ] Document test execution strategies

## Technical Approach

```python
# tests/integration/test_end_to_end_rag_pipeline.py
import pytest

@pytest.mark.integration
def test_full_rag_pipeline_with_real_data():
    """Test complete RAG pipeline from ingestion to query."""
    from mnemonic_cortex.scripts.ingest import ingest_directory
    from mnemonic_cortex.app.main import query_cortex
    
    # 1. Ingest real protocol documents
    ingest_directory("01_PROTOCOLS", max_files=10)
    
    # 2. Query the Cortex
    results = query_cortex("What is the Doctrine of the Infinite Forge?")
    
    # 3. Verify results
    assert len(results) > 0
    assert any("Protocol 78" in str(r) for r in results)
    assert any("Infinite Forge" in str(r) for r in results)

@pytest.mark.integration
def test_council_orchestrator_cortex_integration(tmp_path):
    """Test orchestrator successfully queries Cortex."""
    from council_orchestrator.orchestrator.memory.cortex import CortexInterface
    
    cortex = CortexInterface()
    
    # Test cache warming
    cortex.warm_cache(["chronicles", "protocols"])
    
    # Test query
    results = cortex.query("What is Protocol 101?")
    
    assert len(results) > 0
    assert "Unbreakable Commit" in str(results)
```

```python
# tests/benchmarks/test_rag_query_performance.py
import pytest

@pytest.mark.benchmark
def test_rag_query_latency(benchmark):
    """Benchmark RAG query performance."""
    from mnemonic_cortex.app.main import query_cortex
    
    query = "What is the Mnemonic Cortex?"
    
    # Benchmark the query
    result = benchmark(query_cortex, query)
    
    # Verify result quality
    assert len(result) > 0
    
    # Check performance (baseline: < 500ms)
    assert benchmark.stats.mean < 0.5  # 500ms

@pytest.mark.benchmark
def test_embedding_generation_performance(benchmark):
    """Benchmark embedding generation speed."""
    from nomic import embed
    
    texts = ["Sample text for embedding"] * 100
    
    def generate_embeddings():
        return embed.text(texts, model="nomic-embed-text-v1.5")
    
    result = benchmark(generate_embeddings)
    
    # Calculate throughput
    docs_per_second = 100 / benchmark.stats.mean
    
    # Baseline: > 50 docs/second
    assert docs_per_second > 50
```

```ini
# pytest.ini (project root)
[pytest]
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    benchmark: marks tests as performance benchmarks (deselect with '-m "not benchmark"')
    slow: marks tests as slow (deselect with '-m "not slow"')

# Default: skip slow tests
addopts = -m "not slow and not integration and not benchmark" --strict-markers

# Test discovery
testpaths = tests mnemonic_cortex/tests council_orchestrator/tests forge/OPERATION_PHOENIX_FORGE/tests

# Coverage
[coverage:run]
omit = 
    */tests/*
    */test_*.py
```

```bash
# run_integration_tests.sh
#!/bin/bash
echo "Running integration tests..."
pytest -m integration -v

echo "Running performance benchmarks..."
pytest -m benchmark --benchmark-only
```

## Success Metrics

- [ ] All critical workflows have integration tests
- [ ] Performance baselines established for RAG queries
- [ ] Integration tests pass consistently
- [ ] Performance benchmarks complete in < 2 minutes
- [ ] Clear documentation for running different test types

## Related Protocols

- **P89**: The Clean Forge
- **P101**: The Unbreakable Commit
- **P115**: The Tactical Mandate

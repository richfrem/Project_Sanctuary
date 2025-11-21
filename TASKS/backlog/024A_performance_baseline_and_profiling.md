# Task 024A: Performance Baseline Establishment & Profiling

## Metadata
- **Status**: backlog
- **Priority**: medium
- **Complexity**: medium
- **Category**: performance
- **Estimated Effort**: 4-6 hours
- **Dependencies**: 021A (Mnemonic Cortex tests)
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Parent Task**: 024 (split into 024A, 024B)

## Context

Project Sanctuary lacks performance baselines and systematic profiling. This task establishes measurable performance metrics and identifies bottlenecks.

## Objective

Establish performance baselines for critical operations and implement profiling infrastructure to identify optimization opportunities.

## Acceptance Criteria

### 1. Performance Baseline Establishment
- [ ] Create `tools/performance/establish_baselines.py` script
- [ ] Benchmark RAG query latency (p50, p95, p99)
- [ ] Benchmark embedding generation speed (docs/second)
- [ ] Benchmark vector database query time
- [ ] Benchmark model inference latency (per engine)
- [ ] Save baselines to `baselines/performance_baselines.json`
- [ ] Document baseline methodology

### 2. Profiling Infrastructure
- [ ] Create `tools/performance/profile_rag_pipeline.py`
- [ ] Create `tools/performance/profile_memory_usage.py`
- [ ] Integrate cProfile for CPU profiling
- [ ] Integrate memory_profiler for memory analysis
- [ ] Generate profiling reports (text + visual)
- [ ] Add profiling to development workflow

### 3. Bottleneck Identification
- [ ] Profile RAG query pipeline end-to-end
- [ ] Profile embedding generation
- [ ] Profile vector database operations
- [ ] Identify top 5 performance bottlenecks
- [ ] Document findings in `reports/performance_analysis.md`
- [ ] Prioritize optimization opportunities

### 4. Performance Testing
- [ ] Create `tests/performance/test_baseline_regression.py`
- [ ] Add performance regression detection
- [ ] Configure acceptable performance variance (±10%)
- [ ] Integrate into CI/CD (optional, can run manually)

## Technical Approach

```python
# tools/performance/establish_baselines.py
"""
Establish performance baselines for critical operations.
"""

import time
import statistics
from typing import List, Dict
from pathlib import Path
import json
import platform
import sys

def benchmark_rag_query(queries: List[str], iterations: int = 100) -> Dict:
    """Benchmark RAG query performance."""
    from mnemonic_cortex.app.main import query_cortex
    
    latencies = []
    
    for query in queries:
        for _ in range(iterations):
            start = time.perf_counter()
            results = query_cortex(query, k=5)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    return {
        "operation": "rag_query",
        "iterations": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": statistics.quantiles(latencies, n=20)[18],
        "p99_ms": statistics.quantiles(latencies, n=100)[98],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "stdev_ms": statistics.stdev(latencies)
    }

def benchmark_embedding_generation(batch_sizes: List[int] = [1, 10, 50]) -> Dict:
    """Benchmark embedding generation at different batch sizes."""
    from nomic import embed
    
    test_texts = ["Sample text for embedding testing"] * 100
    results = {}
    
    for batch_size in batch_sizes:
        latencies = []
        
        for i in range(0, len(test_texts), batch_size):
            batch = test_texts[i:i+batch_size]
            start = time.perf_counter()
            embeddings = embed.text(batch, model="nomic-embed-text-v1.5")
            end = time.perf_counter()
            
            docs_per_second = len(batch) / (end - start)
            latencies.append(docs_per_second)
        
        results[f"batch_{batch_size}"] = {
            "mean_docs_per_second": statistics.mean(latencies),
            "median_docs_per_second": statistics.median(latencies)
        }
    
    return results

def save_baselines(baselines: Dict):
    """Save baseline metrics with metadata."""
    output_path = Path("baselines/performance_baselines.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    baselines["metadata"] = {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count()
    }
    
    output_path.write_text(json.dumps(baselines, indent=2))
    print(f"✓ Baselines saved to {output_path}")

if __name__ == "__main__":
    print("Establishing performance baselines...")
    
    baselines = {}
    
    # RAG query benchmark
    test_queries = [
        "What is the Doctrine of the Infinite Forge?",
        "How does the Mnemonic Cortex work?",
        "What is Protocol 101?"
    ]
    print("Benchmarking RAG queries...")
    baselines["rag_query"] = benchmark_rag_query(test_queries)
    
    # Embedding benchmark
    print("Benchmarking embedding generation...")
    baselines["embedding_generation"] = benchmark_embedding_generation()
    
    # Save results
    save_baselines(baselines)
    
    # Print summary
    print("\n=== Performance Baselines ===")
    print(f"RAG Query - Median: {baselines['rag_query']['median_ms']:.2f}ms")
    print(f"RAG Query - P95: {baselines['rag_query']['p95_ms']:.2f}ms")
    print(f"RAG Query - P99: {baselines['rag_query']['p99_ms']:.2f}ms")
```

```python
# tools/performance/profile_rag_pipeline.py
"""
Detailed profiling of RAG pipeline.
"""

import cProfile
import pstats
from io import StringIO
from pathlib import Path

def profile_rag_pipeline(query: str = "What is Protocol 78?"):
    """Profile complete RAG pipeline execution."""
    profiler = cProfile.Profile()
    
    # Profile the query
    profiler.enable()
    from mnemonic_cortex.app.main import query_cortex
    results = query_cortex(query)
    profiler.disable()
    
    # Save profile data
    output_dir = Path("profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Text report
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(50)  # Top 50 functions
    
    (output_dir / "rag_profile.txt").write_text(s.getvalue())
    
    # Binary profile for visualization
    profiler.dump_stats(str(output_dir / "rag_profile.prof"))
    
    print(f"✓ Profile saved to {output_dir}/")
    print("  View with: snakeviz profiles/rag_profile.prof")
    
    return results

if __name__ == "__main__":
    profile_rag_pipeline()
```

## Success Metrics

- [ ] Performance baselines established for all critical operations
- [ ] Profiling reports generated and analyzed
- [ ] Top 5 bottlenecks identified and documented
- [ ] Performance regression tests created
- [ ] Baseline methodology documented

## Related Protocols

- **P85**: The Mnemonic Cortex Protocol
- **P89**: The Clean Forge

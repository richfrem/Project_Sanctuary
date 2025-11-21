# Task 024B: Performance Optimization & Monitoring

## Metadata
- **Status**: backlog
- **Priority**: medium
- **Complexity**: medium
- **Category**: performance
- **Estimated Effort**: 4-6 hours
- **Dependencies**: 024A
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Parent Task**: 024 (split into 024A, 024B)

## Context

After establishing baselines and identifying bottlenecks (Task 024A), this task implements optimizations and sets up real-time monitoring.

## Objective

Implement performance optimizations for identified bottlenecks and establish real-time performance monitoring infrastructure.

## Acceptance Criteria

### 1. RAG Pipeline Optimization
- [ ] Implement query result caching (LRU cache)
- [ ] Optimize embedding batch processing
- [ ] Add query preprocessing/normalization
- [ ] Implement parallel document processing (if beneficial)
- [ ] Benchmark before/after optimization
- [ ] Document optimization techniques used

### 2. Resource Monitoring
- [ ] Create `tools/performance/monitor.py` daemon
- [ ] Track CPU usage per module
- [ ] Track memory usage and detect leaks
- [ ] Track disk I/O operations
- [ ] Generate resource usage reports
- [ ] Add lightweight metrics collection

### 3. Performance Improvements
- [ ] Achieve 20%+ improvement in RAG query latency
- [ ] Reduce memory usage for long-running operations
- [ ] Optimize batch sizes for embedding generation
- [ ] Implement connection pooling where applicable
- [ ] Add warmup for first query (model loading)

### 4. Monitoring Dashboard (Optional)
- [ ] Set up basic Prometheus metrics export (optional)
- [ ] Create simple performance dashboard script
- [ ] Add alerting for performance degradation (optional)
- [ ] Document monitoring setup

## Technical Approach

```python
# mnemonic_cortex/core/optimized_vector_db_service.py
"""
Optimized vector database service with caching.
"""

from functools import lru_cache
import hashlib
from typing import List, Dict
import json

class OptimizedVectorDBService:
    """Enhanced vector DB service with performance optimizations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._warmup_complete = False
    
    @lru_cache(maxsize=1000)
    def _cached_query(self, query_hash: str, k: int) -> str:
        """
        LRU cache for query results.
        
        Returns JSON string for hashability.
        """
        results = self._execute_query(query_hash, k)
        return json.dumps(results)
    
    def query(self, query: str, k: int = 5) -> List[Dict]:
        """
        Query with automatic caching.
        
        Optimization: Identical queries return cached results.
        """
        # Normalize query
        normalized_query = query.strip().lower()
        
        # Create hash for cache key
        query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()
        
        # Check cache
        cached_result = self._cached_query(query_hash, k)
        return json.loads(cached_result)
    
    def warmup(self):
        """Warmup models and caches for first query."""
        if not self._warmup_complete:
            # Load models, initialize connections
            self._execute_query("warmup", k=1)
            self._warmup_complete = True
    
    def batch_embed(self, texts: List[str], batch_size: int = 50) -> List:
        """
        Optimized batch embedding generation.
        
        Optimization: Process in optimal batch sizes.
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._generate_embeddings(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
```

```python
# tools/performance/monitor.py
"""
Lightweight performance monitoring.
"""

import time
import psutil
from pathlib import Path
import json

class PerformanceMonitor:
    """Simple performance monitoring."""
    
    def __init__(self, output_file: str = "logs/performance_metrics.jsonl"):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def collect_metrics(self) -> Dict:
        """Collect current system metrics."""
        return {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
        }
    
    def log_metrics(self):
        """Log metrics to file."""
        metrics = self.collect_metrics()
        
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def start_monitoring(self, interval: int = 60):
        """Start monitoring loop."""
        print(f"Performance monitoring started (interval: {interval}s)")
        print(f"Logging to: {self.output_file}")
        
        try:
            while True:
                self.log_metrics()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.start_monitoring(interval=60)  # Log every minute
```

```python
# tests/performance/test_optimization_improvements.py
"""
Test that optimizations achieve target improvements.
"""

import pytest
import json
from pathlib import Path

def load_baseline(metric_name: str) -> float:
    """Load baseline metric from file."""
    baseline_file = Path("baselines/performance_baselines.json")
    baselines = json.loads(baseline_file.read_text())
    return baselines["rag_query"][metric_name]

@pytest.mark.performance
def test_rag_query_improvement():
    """Test that RAG query latency improved by 20%."""
    from mnemonic_cortex.app.main import query_cortex
    import time
    
    # Get baseline
    baseline_p95 = load_baseline("p95_ms")
    
    # Benchmark current performance
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        query_cortex("What is Protocol 78?")
        latencies.append((time.perf_counter() - start) * 1000)
    
    current_p95 = sorted(latencies)[94]  # 95th percentile
    
    # Check for 20% improvement
    improvement = (baseline_p95 - current_p95) / baseline_p95
    
    assert improvement >= 0.20, f"Only {improvement*100:.1f}% improvement, target is 20%"
```

## Success Metrics

- [ ] 20%+ improvement in RAG query p95 latency
- [ ] Memory usage stable over 1000+ queries
- [ ] Optimizations documented with before/after metrics
- [ ] Resource monitoring operational
- [ ] Performance regression tests passing

## Related Protocols

- **P85**: The Mnemonic Cortex Protocol
- **P89**: The Clean Forge
- **P97**: Generative Development Cycle

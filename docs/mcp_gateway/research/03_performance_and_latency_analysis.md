# Performance and Latency Analysis

**Research Date:** 2025-12-15  
**Task:** 110 - Dynamic MCP Gateway Architecture  
**Focus:** API Gateway performance benchmarks, latency overhead, optimization strategies

---

## Executive Summary

Research on API Gateway performance reveals:
- **Typical Latency Overhead:** 10-50ms for well-optimized gateways
- **High-Performance Gateways:** Kong (137,000 RPS, 3.82ms p95), Apache APISIX (18,000 QPS, 0.2ms)
- **Sanctuary Estimate:** 15-30ms total overhead (acceptable for human-in-loop workflows)

**Key Finding:** The benefits of centralized management, security, and scalability outweigh the latency cost for our use case.

---

## 1. API Gateway Latency Fundamentals

### 1.1 What Contributes to Latency?

**Network Hops:**
- Client → Gateway: 1-5ms (local)
- Gateway → Backend: 5-15ms (container network)
- Total round-trip: 10-30ms

**SSL/TLS Handshakes:**
- Initial connection: 50-100ms
- Resumed connection: 5-10ms
- **Mitigation:** Connection pooling, keep-alive

**Gateway Processing:**
- Authentication/Authorization: 1-5ms
- Rate limiting: 0.5-2ms
- Logging: 0.5-1ms
- Routing logic: 0.5-2ms
- **Total:** 3-10ms

**Backend Integration:**
- JSON-RPC serialization: 0.5-1ms
- HTTP request/response: 5-15ms
- Backend processing: Variable (10-1000ms+)

### 1.2 Total Latency Budget

**Sanctuary Gateway Estimate:**
```
User Request → Claude Desktop: 0ms (local)
Claude → Gateway (stdio): 1-2ms
Gateway Processing: 5-10ms
Gateway → Backend (HTTP): 10-20ms
Backend Processing: 50-500ms (varies by tool)
Backend → Gateway: 10-20ms
Gateway → Claude: 1-2ms
Claude → User: 0ms (local)
───────────────────────────────────
Total Overhead: 27-54ms
Backend Processing: 50-500ms
───────────────────────────────────
Total User-Facing: 77-554ms
```

**Acceptable?** YES - Human-in-loop workflows tolerate 100-500ms latency easily.

---

## 2. Production Gateway Benchmarks

### 2.1 Kong Gateway 3.6

**Performance:**
- **RPS:** 137,000+ requests per second
- **Latency (p50):** 1.2ms
- **Latency (p95):** 3.82ms
- **Latency (p99):** 8.5ms

**Configuration:**
- Basic proxy (no plugins)
- Single-core CPU
- Local network

**Relevance to Sanctuary:**
- Kong is **overkill** for our scale (12 servers, <100 RPS expected)
- But proves that gateway overhead can be <5ms with optimization

### 2.2 Apache APISIX

**Performance:**
- **QPS:** 18,000 queries per second (with plugins enabled)
- **Latency:** 0.2ms average
- **Single-core CPU test**

**Features:**
- Lua-based (extremely fast)
- Built on OpenResty/NGINX
- Low memory footprint

**Relevance to Sanctuary:**
- Also overkill, but validates low-latency gateway pattern
- Python-based gateway will be slower, but still acceptable

### 2.3 Benchmark Comparison

| Gateway | RPS | p50 Latency | p95 Latency | Language |
|---------|-----|-------------|-------------|----------|
| Kong Enterprise | 54,250 | <10ms | <30ms | Lua/C |
| Apache APISIX | 18,000 | 0.2ms | N/A | Lua |
| Kong Gateway 3.6 | 137,000 | 1.2ms | 3.82ms | Lua/C |
| Google Apigee X | ~40,000 | <20ms | <50ms | Java |
| **Sanctuary (Est.)** | **<100** | **15ms** | **30ms** | **Python** |

**Conclusion:** Even a Python-based gateway should achieve <30ms latency, well within acceptable range.

---

## 3. Latency Mitigation Strategies

### 3.1 Connection Pooling

**Problem:** Creating new HTTP connections is expensive (SSL handshake, TCP setup)

**Solution:** Reuse connections to backend servers

```python
import httpx

class GatewayProxy:
    def __init__(self):
        # Connection pool with keep-alive
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0
            ),
            timeout=httpx.Timeout(10.0)
        )
    
    async def proxy(self, endpoint: str, request: dict):
        # Reuses existing connection if available
        response = await self.client.post(endpoint, json=request)
        return response.json()
```

**Impact:** Reduces per-request latency by 10-50ms (eliminates handshake)

### 3.2 Caching

**Problem:** Repeated queries to backend servers waste time

**Solution:** Cache responses for frequently accessed data

```python
from functools import lru_cache
import hashlib
import json

class CachingProxy:
    def __init__(self):
        self.cache = {}  # In-memory cache
        self.ttl = 300  # 5 minutes
    
    async def proxy_with_cache(self, tool_name: str, params: dict):
        # Generate cache key
        cache_key = hashlib.sha256(
            f"{tool_name}:{json.dumps(params, sort_keys=True)}".encode()
        ).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            cached, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                return cached
        
        # Cache miss - fetch from backend
        result = await self.proxy(tool_name, params)
        self.cache[cache_key] = (result, time.time())
        return result
```

**Impact:** 
- Cache hit: 0.5-1ms (memory lookup)
- Cache miss: Normal latency
- **For read-heavy workloads:** 50-90% latency reduction

**Caveats:**
- Only cache idempotent operations (queries, not mutations)
- Implement cache invalidation for data updates
- Monitor cache hit rate

### 3.3 Async I/O

**Problem:** Blocking I/O wastes CPU cycles waiting for network

**Solution:** Use async/await for concurrent request handling

```python
import asyncio

class AsyncGateway:
    async def handle_multiple_requests(self, requests: list):
        # Process all requests concurrently
        tasks = [self.proxy(req) for req in requests]
        results = await asyncio.gather(*tasks)
        return results
```

**Impact:**
- Single request: No improvement
- Multiple concurrent requests: 2-10x throughput improvement
- **For Sanctuary:** Minimal benefit (typically 1 request at a time from Claude)

### 3.4 Load Balancing

**Problem:** Single backend instance becomes bottleneck

**Solution:** Distribute requests across multiple instances

```python
class LoadBalancingProxy:
    def __init__(self):
        self.backends = {
            "rag_cortex": [
                "http://localhost:8001",
                "http://localhost:8002",  # Second instance
            ]
        }
        self.current_index = {}
    
    def get_backend(self, server_name: str) -> str:
        """Round-robin load balancing."""
        endpoints = self.backends[server_name]
        idx = self.current_index.get(server_name, 0)
        endpoint = endpoints[idx % len(endpoints)]
        self.current_index[server_name] = idx + 1
        return endpoint
```

**Impact:**
- Distributes load across instances
- Prevents single-instance saturation
- **For Sanctuary:** Not needed initially (low traffic)

---

## 4. Sanctuary-Specific Optimizations

### 4.1 Warm Connections

**Strategy:** Keep connections to frequently used servers warm

```python
class WarmConnectionManager:
    def __init__(self):
        self.warm_servers = ["rag_cortex", "git_workflow", "task"]
        self.client = httpx.AsyncClient()
    
    async def warmup(self):
        """Pre-establish connections to frequently used servers."""
        for server in self.warm_servers:
            endpoint = registry.get_endpoint(server)
            # Send health check to establish connection
            await self.client.get(f"{endpoint}/health")
```

**Impact:** First request to warm server: 5-10ms faster

### 4.2 Lazy Backend Initialization

**Strategy:** Only start backend servers when first requested

```python
class LazyBackendManager:
    def __init__(self):
        self.running_servers = set()
    
    async def ensure_running(self, server_name: str):
        if server_name not in self.running_servers:
            # Start container
            subprocess.run(["podman", "start", f"{server_name}-mcp"])
            # Wait for health check
            await self.wait_for_health(server_name)
            self.running_servers.add(server_name)
```

**Impact:**
- Reduces resource usage (only run needed servers)
- First request: +2-5s (container startup)
- Subsequent requests: Normal latency

**Trade-off:** Acceptable for infrequently used servers (e.g., `orchestrator`)

### 4.3 Circuit Breaker Pattern

**Strategy:** Fail fast if backend is down

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = {}
        self.threshold = failure_threshold
        self.timeout = timeout
        self.open_until = {}
    
    async def call(self, server_name: str, func):
        # Check if circuit is open
        if server_name in self.open_until:
            if time.time() < self.open_until[server_name]:
                raise Exception(f"Circuit breaker open for {server_name}")
            else:
                # Try to close circuit
                del self.open_until[server_name]
                self.failure_count[server_name] = 0
        
        try:
            result = await func()
            self.failure_count[server_name] = 0
            return result
        except Exception as e:
            self.failure_count[server_name] = self.failure_count.get(server_name, 0) + 1
            if self.failure_count[server_name] >= self.threshold:
                # Open circuit
                self.open_until[server_name] = time.time() + self.timeout
            raise e
```

**Impact:**
- Failed server: Immediate error (no 10s timeout)
- User experience: Faster failure feedback

---

## 5. Performance Monitoring

### 5.1 Metrics to Track

**Latency Metrics:**
- Gateway processing time (p50, p95, p99)
- Backend response time (per server)
- Total request time (end-to-end)

**Throughput Metrics:**
- Requests per second (RPS)
- Requests per minute (RPM)
- Concurrent requests

**Error Metrics:**
- Error rate (%)
- Timeout rate (%)
- Circuit breaker trips

### 5.2 Implementation

```python
import time
from prometheus_client import Histogram, Counter

# Metrics
request_latency = Histogram(
    'gateway_request_latency_seconds',
    'Gateway request latency',
    ['tool_name', 'server_name']
)

request_count = Counter(
    'gateway_requests_total',
    'Total gateway requests',
    ['tool_name', 'server_name', 'status']
)

class MonitoredProxy:
    async def proxy(self, tool_name: str, params: dict):
        server_name = registry.route(tool_name)
        start = time.time()
        
        try:
            result = await self._do_proxy(server_name, tool_name, params)
            request_count.labels(tool_name, server_name, 'success').inc()
            return result
        except Exception as e:
            request_count.labels(tool_name, server_name, 'error').inc()
            raise e
        finally:
            latency = time.time() - start
            request_latency.labels(tool_name, server_name).observe(latency)
```

### 5.3 Alerting Thresholds

**Latency Alerts:**
- p95 > 100ms: Warning
- p95 > 500ms: Critical

**Error Alerts:**
- Error rate > 5%: Warning
- Error rate > 20%: Critical

**Availability Alerts:**
- Server down > 1 minute: Critical

---

## 6. Comparison: With vs Without Gateway

### 6.1 Current (Static Loading)

**Latency:**
- User → Claude → MCP Server: 5-10ms
- MCP Server Processing: 50-500ms
- **Total:** 55-510ms

**Context Overhead:**
- 12 servers × 8 tools × 80 tokens = **7,680 tokens**

### 6.2 Proposed (Dynamic Gateway)

**Latency:**
- User → Claude → Gateway: 5-10ms
- Gateway Routing: 5-10ms
- Gateway → Backend: 10-20ms
- Backend Processing: 50-500ms
- **Total:** 70-540ms

**Context Overhead:**
- 1 gateway × 10 meta-tools × 100 tokens = **1,000 tokens**

### 6.3 Trade-off Analysis

| Metric | Current | Proposed | Change |
|--------|---------|----------|--------|
| Latency (p50) | 100ms | 120ms | +20ms (+20%) |
| Latency (p95) | 300ms | 350ms | +50ms (+17%) |
| Context Overhead | 7,680 tokens | 1,000 tokens | -6,680 (-87%) |
| Scalability | 20 servers max | 100+ servers | 5x improvement |
| Flexibility | Static config | Dynamic discovery | Qualitative |

**Verdict:** **20ms latency cost is acceptable for 87% context savings and 5x scalability**

---

## 7. Recommendations

### 7.1 Immediate (Phase 1)

1. **Use Connection Pooling:** httpx.AsyncClient with keep-alive
2. **Implement Basic Monitoring:** Log latency for each request
3. **Set Timeout:** 10s timeout for backend requests

### 7.2 Short-Term (Phase 2)

1. **Add Caching:** Cache read-only operations (cortex_query, protocol_get)
2. **Implement Circuit Breaker:** Fail fast for down servers
3. **Add Prometheus Metrics:** Track p50/p95/p99 latency

### 7.3 Long-Term (Phase 3)

1. **Optimize Hot Paths:** Profile and optimize most-used tools
2. **Consider Rust/Go Rewrite:** If latency becomes critical (unlikely)
3. **Add Load Balancing:** If traffic exceeds single-instance capacity

---

## 8. References

### Benchmarks
- Kong Gateway 3.6 Performance Report
- Apache APISIX Benchmark Results
- Gigaom API Gateway Comparison (Kong vs Apigee vs MuleSoft)

### Optimization Strategies
- Tyk: API Gateway Latency Optimization
- Solo.io: Envoy Proxy Performance
- Microsoft: API Gateway Patterns

### Related Sanctuary Documents
- Protocol 116: Ollama Container Network
- Task 087: MCP Operations Testing

---

## Conclusion

**Gateway Latency Overhead:** 15-30ms (acceptable)

**Optimization Priority:**
1. Connection pooling (easy, high impact)
2. Caching (medium effort, high impact for reads)
3. Circuit breaker (easy, improves failure handling)
4. Monitoring (essential for visibility)

**Bottom Line:** The 20-50ms latency cost is negligible compared to the benefits of context efficiency, scalability, and centralized management.

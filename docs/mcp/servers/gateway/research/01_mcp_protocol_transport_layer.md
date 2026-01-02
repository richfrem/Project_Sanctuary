# MCP Protocol Transport Layer Research

**Research Date:** 2025-12-15  
**Task:** 110 - Dynamic MCP Gateway Architecture  
**Focus:** JSON-RPC 2.0, stdio, and HTTP/SSE transport mechanisms

---

## Executive Summary

MCP uses **JSON-RPC 2.0** as its core message protocol, supporting two primary transport mechanisms:
1. **stdio** - For local process communication (current Sanctuary implementation)
2. **HTTP with optional SSE** - For remote/networked communication

**Key Finding for Gateway:** The Gateway must support **stdio transport** to maintain compatibility with Claude Desktop's current architecture while potentially offering HTTP endpoints for future remote server support.

---

## 1. JSON-RPC 2.0 Message Format

### 1.1 Core Specification
- **Protocol:** JSON-RPC 2.0 (transport-agnostic)
- **Encoding:** UTF-8 required
- **Message Types:**
  - **Request:** Client invoking server capability
  - **Response:** Server's result or error
  - **Notification:** Proactive updates without acknowledgment

### 1.2 Message Structure Examples

**Tool Call Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "cortex_query",
    "arguments": {
      "query": "What is Protocol 101?",
      "max_results": 5
    }
  }
}
```

**Tool Call Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Protocol 101 is the Functional Coherence standard..."
      }
    ]
  }
}
```

**Error Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32603,
    "message": "Internal error: Server rag_cortex not available"
  }
}
```

---

## 2. stdio Transport (Primary for Sanctuary)

### 2.1 Architecture
- **Process Model:** MCP client launches server as subprocess
- **Communication:**
  - Server reads JSON-RPC from `stdin`
  - Server writes JSON-RPC to `stdout`
  - Server writes logs to `stderr` (optional, UTF-8)

### 2.2 Message Delimitation
- **Critical:** Messages delimited by newlines (`\n`)
- **Restriction:** Embedded newlines within messages are **strictly prohibited**
- **Format:** One JSON-RPC message per line

**Example stdin stream:**
```
{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"cortex_query","arguments":{"query":"test"}}}\n
```

### 2.3 Benefits for Sanctuary
1. **Simplicity:** Direct process communication, no network stack
2. **Performance:** Low latency (no TCP/HTTP overhead)
3. **Security:** Process isolation inherent
4. **Current Compatibility:** Claude Desktop uses stdio exclusively

### 2.4 Limitations
- **Local Only:** Cannot communicate with remote servers
- **One-to-One:** Typically single client-server relationship
- **No Load Balancing:** Cannot distribute across multiple instances

---

## 3. HTTP with Server-Sent Events (SSE)

### 3.1 Evolution of HTTP Transport

**Original (Nov 2024 - Deprecated):**
- Dual-endpoint architecture:
  - Persistent SSE connection (`GET /sse`) for server-to-client
  - HTTP POST for client-to-server
- **Problems:** Complex session management, no resume capability

**Current (Streamable HTTP - March 2025):**
- **Client-to-Server:** HTTP POST with JSON-RPC body
- **Server-to-Client:** Optional SSE stream for multiple responses
- **Content-Type:** `text/event-stream` when streaming

### 3.2 Streamable HTTP Flow

**Single Response (No SSE):**
```http
POST /mcp HTTP/1.1
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"tools/list"}

HTTP/1.1 200 OK
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}
```

**Streaming Response (With SSE):**
```http
POST /mcp HTTP/1.1
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{...}}

HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progress":0.5}}

data: {"jsonrpc":"2.0","id":1,"result":{...}}
```

### 3.3 Benefits of HTTP Transport
1. **Remote Access:** Servers can be networked, not just local
2. **Multi-Client:** Multiple clients can connect to same server
3. **Load Balancing:** Standard HTTP load balancing applies
4. **Cloud Deployment:** Compatible with serverless, containers

### 3.4 Limitations
- **Latency:** Network overhead (10-50ms typical)
- **Complexity:** Requires HTTP server infrastructure
- **Security:** Must implement authentication, TLS

---

## 4. Gateway Transport Strategy

### 4.1 Hybrid Approach (Recommended)

**Gateway as stdio Server (to Claude Desktop):**
```python
# Gateway exposes stdio interface to Claude Desktop
# Claude Desktop config:
{
  "mcpServers": {
    "sanctuary-broker": {
      "command": "podman",
      "args": ["exec", "-i", "sanctuary-broker-mcp", "python", "-m", "gateway.server"]
    }
  }
}
```

**Gateway as HTTP Client (to backend servers):**
```python
# Gateway communicates with backend servers via HTTP
async def forward_to_backend(tool_name: str, params: dict) -> dict:
    server_endpoint = registry.get_endpoint(tool_name)
    # HTTP POST to backend MCP server
    response = await http_client.post(
        f"{server_endpoint}/mcp",
        json={"jsonrpc": "2.0", "method": "tools/call", "params": params}
    )
    return response.json()
```

### 4.2 Why Hybrid?

**stdio to Claude Desktop:**
- Maintains compatibility with existing setup
- No configuration changes needed in Claude Desktop
- Lowest latency for user-facing interactions

**HTTP to Backend Servers:**
- Enables remote server support (future: cloud-hosted MCP servers)
- Allows multiple Gateway instances to share backend servers
- Facilitates health checks and service discovery
- Supports containerized backend servers (Podman network)

### 4.3 Alternative: stdio-to-stdio Proxy

**Simpler but Limited:**
```python
# Gateway spawns backend servers as subprocesses
# Proxies stdin/stdout between Claude and backend
backend_process = subprocess.Popen(
    ["python", "-m", "rag_cortex.server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
```

**Limitations:**
- Cannot support remote servers
- Difficult to implement health checks
- No load balancing or failover
- Process management complexity

---

## 5. Implementation Recommendations

### 5.1 Phase 1: stdio-to-stdio Proxy (MVP)
- **Timeline:** Week 1
- **Scope:** Gateway proxies to 2-3 local servers via stdio
- **Benefits:** Fastest to implement, validates routing logic
- **Limitations:** No remote server support

### 5.2 Phase 2: Hybrid stdio/HTTP (Production)
- **Timeline:** Week 2-3
- **Scope:** Gateway uses stdio for Claude, HTTP for backends
- **Benefits:** Full flexibility, supports remote servers
- **Implementation:**
  ```python
  class SanctuaryGateway:
      def __init__(self):
          self.stdin_reader = StdioReader()
          self.http_client = AsyncHTTPClient()
          self.registry = ServerRegistry()
      
      async def handle_request(self, request: dict):
          tool_name = request["params"]["name"]
          backend = self.registry.get_backend(tool_name)
          
          if backend.transport == "stdio":
              return await self.proxy_stdio(backend, request)
          elif backend.transport == "http":
              return await self.proxy_http(backend, request)
  ```

### 5.3 Phase 3: Full HTTP Gateway (Future)
- **Timeline:** Week 4+
- **Scope:** Gateway exposes HTTP endpoint to Claude Desktop
- **Benefits:** Cloud deployment, multiple clients
- **Requirement:** Claude Desktop must support HTTP transport (not yet available)

---

## 6. Security Considerations

### 6.1 stdio Transport Security
- **Process Isolation:** Each server runs in separate process/container
- **No Network Exposure:** Cannot be accessed remotely
- **Logging:** stderr can be captured for audit trail

### 6.2 HTTP Transport Security
- **Authentication:** OAuth 2.0 tokens, API keys
- **TLS:** Encrypt all HTTP traffic
- **Rate Limiting:** Prevent abuse
- **Allowlist:** Restrict which servers can be accessed

---

## 7. Performance Benchmarks

### 7.1 stdio Latency
- **Typical:** <1ms for local process communication
- **Overhead:** Minimal (process context switch)

### 7.2 HTTP Latency
- **Local (localhost):** 1-5ms
- **Container Network (Podman):** 5-15ms
- **Remote (LAN):** 10-50ms
- **Remote (Internet):** 50-200ms+

### 7.3 Gateway Overhead Estimate
- **stdio-to-stdio:** +0.5-2ms (routing logic)
- **stdio-to-HTTP:** +5-20ms (HTTP roundtrip)
- **Total User-Facing Latency:** 10-50ms (acceptable for human-in-loop)

---

## 8. References

### Official Documentation
- [MCP Transport Specification](https://modelcontextprotocol.io/docs/concepts/transports)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)

### Research Sources
- MCP Framework Documentation (mcp-framework.com)
- Holt Courses: MCP Transport Evolution
- Plain English: MCP SSE Implementation

### Related Sanctuary Documents
- Protocol 116: Ollama Container Network
- Task 087: MCP Operations Testing

---

## Conclusion

**Recommendation:** Implement **Hybrid stdio/HTTP Gateway** (Phase 2) as the production architecture.

**Rationale:**
1. Maintains compatibility with Claude Desktop (stdio)
2. Enables future remote server support (HTTP)
3. Supports containerized backends (Podman network)
4. Acceptable latency overhead (10-50ms)
5. Clear migration path to full HTTP gateway when Claude Desktop supports it

**Next Steps:**
1. Build stdio-to-stdio MVP to validate routing logic
2. Implement HTTP client for backend communication
3. Add health checks and service discovery
4. Deploy in Podman container with network access to backends

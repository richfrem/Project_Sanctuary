# Gateway Integration Testing: Modular Architecture Proposal

## 1. Problem Statement
The current integration test suite (`tests/mcp_servers/gateway/integration/*.py`) suffers from high logical redundancy. 
- **Duplication**: Every file repeats `httpx` connection logic, token management, and JSON-RPC payload assembly.
- **Maintenance Fragility**: Changes to the Gateway API or security protocols require manual updates across 8+ files.
- **Inconsistency**: Discrepancies exist between how the CLI manager (`gateway_client.py`) and the unit tests call the same RPC endpoints.

## 2. Core Principles
- **DRY (Don't Repeat Yourself)**: All Gateway RPC logic must reside in a single class.
- **Separation of Concerns**: Test files should focus on *what* to test (tool names/arguments), not *how* to connect (HTTP/RPC details).
- **Tool Agnostic**: The core client should not know about specific tools; it should be a generic transport layer.

## 3. Proposed Architecture

### A. The Engine: `gateway_test_client.py`
Create a central utility class to handle the "heavy lifting":
- `.env` and `GatewayConfig` management.
- Persistent `requests.Session` or `httpx.Client` with retries.
- `execute_rpc(method, params)` core function.
- `list_tools()` wrapper.

## 3. File Rationalization Matrix
We are consolidating the following legacy/redundant files into the modular architecture:

| File | Context | Current Issue | Transition Path |
| :--- | :--- | :--- | :--- |
| `test_cortex_gateway.py` | Tier 2/3 Mix | Talks directly to port 8104. | Refactor to T3 using `GatewayTestClient`. |
| `test_domain_gateway.py` | Tier 2/3 Mix | Large, redundant HTTP calls. | Strip HTTP logic; use `GatewayTestClient`. |
| `test_filesystem_gateway.py`| Tier 2/3 Mix | Redundant payload assembly. | Use `GatewayTestClient` for all RPC. |
| `test_git_gateway.py` | Tier 2/3 Mix | Redundant session logic. | Use `GatewayTestClient`. |
| `test_network_gateway.py` | Tier 2/3 Mix | Redundant error handling. | Use `GatewayTestClient`. |
| `test_utils_gateway.py` | Tier 2/3 Mix | Redundant SSE checks. | Focus on T3 Bridge verification. |
| `test_gateway_blackbox.py` | Infra Check | Uses separate `requests` logic. | Use `GatewayTestClient` config for pulse. |

## 4. Proposed Architecture

### A. The Engine: [gateway_test_client.py](../lib/gateway_test_client.py)
A class-based modular core that handles:
- **Environment**: Automatic `.env` discovery.
- **Session**: Persistent `requests.Session` with retries and SSL bypass for local testing.
- **Authentication**: Centralized Bearer Token injection.
- **RPC Logic**: Canonical `call()` and `list_tools()` implementation.

### B. Refactored Integration Tests
Individual cluster tests will no longer manage HTTP details. They will simply specify payloads:

```python
from gateway_test_client import GatewayTestClient

def test_adr_list():
    client = GatewayTestClient()
    res = client.call("sanctuary_domain-adr-list", {})
    assert res["success"] is True
```

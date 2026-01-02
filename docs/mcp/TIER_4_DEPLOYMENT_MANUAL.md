# Tier 4: Zero-Trust Deployment Manual (The Ideal State)

**Objective:** Final verification that tools are visible and functional in the actual IDE (Antigravity/Claude). This is the "Absolute Reality" check.

> **Status:** 🚀 Living Standard
> **Prerequisites:** Tier 1 (Unit), Tier 2 (Integration), and Tier 3 (E2E) must pass.

## 🏛️ System Architecture

![gateway_testing_architecture](docs/architecture_diagrams/system/gateway_testing_architecture.png)

*[Source: gateway_testing_architecture.mmd](docs/architecture_diagrams/system/gateway_testing_architecture.mmd)*

---

## 1. ⚙️ Configuration (IDE Setup)

To verify the fleet in "Direct Connect" mode (or via Gateway bridge), update your local `mcp_config.json` (usually `~/.gemini/antigravity/mcp_config.json` or similar for Claude Desktop).

### Production-Ready SSE Endpoints
Add/Update the following server checks to confirm direct availability on the `sanctuary-net` host mapping (localhost):

```json
{
  "mcpServers": {
    "sanctuary_utils": {
      "command": "python3",
      "args": ["-m", "mcp_servers.gateway.clusters.sanctuary_utils.server"],
      "env": { "MCP_TRANSPORT": "sse" },
      "url": "http://localhost:8100/sse"
    },
    "sanctuary_filesystem": {
      "url": "http://localhost:8101/sse"
    },
    "sanctuary_network": {
      "url": "http://localhost:8102/sse"
    },
    "sanctuary_git": {
      "url": "http://localhost:8103/sse"
    },
    "sanctuary_cortex": {
      "url": "http://localhost:8104/sse"
    },
    "sanctuary_domain": {
      "url": "http://localhost:8105/sse"
    }
  }
}
```
*Note: The above JSON is ensuring the IDE connects via SSE. Adjust generic 'command' based setups if using stdio bridge for local testing.*

---

## 2. 🔍 Discovery Check (Metadata Verification)

Once configured and IDE restarted, open "Search and tools" (or equivalent UI) and verify:

### Checklist for ADR 076 Compliance
- [ ] **Sanctuary Utils**: `time-get-current-time`, `calculator-add` visible?
- [ ] **Sanctuary FS**: `code-read`, `code-list-files` visible?
- [ ] **Sanctuary Network**: `fetch-url` visible?
- [ ] **Sanctuary Git**: `git-status` visible?
- [ ] **Sanctuary Cortex**: `cortex-query` visible?
- [ ] **Sanctuary Domain**: `chronicle-list-entries` visible?

> **Failure Condition:** If tools are missing, check `docker-compose logs` for `server.py` crash or 404s.

---

## 3. 🔄 Verification Loop (Real World Execution)

Execute one "Real World" parameter test for each cluster to confirm end-to-end functionality.

### Template: Verification Log

| Cluster | Tool | Test Parameters | Expected Outcome | Pass? |
| :--- | :--- | :--- | :--- | :--- |
| **utils** | `time-get-current-time` | `{"timezone_name": "UTC"}` | Returns timestamp string | [ ] |
| **filesystem** | `code-read` | `{"path": "/app/README.md"}` | Returns file content | [ ] |
| **network** | `fetch-url` | `{"url": "https://example.com"}` | Returns HTML content | [ ] |
| **git** | `git-status` | `{}` | Returns branch/diff status | [ ] |
| **cortex** | `cortex-query` | `{"query": "Protocol 101"}` | Returns RAG results | [ ] |
| **domain** | `task-list-tasks` | `{}` | Returns JSON list of tasks | [ ] |

---

## 4. 🚑 Troubleshooting

- **Tools not showing?** Restart IDE. Check `mcp_config.json` JSON syntax.
- **Connection Refused?** Ensure Podman containers are UP (`podman ps`).
- **Timeout?** Check Gateway/Container logs. Use Tier 3 verification script.

**Sign-off:**
_Verified by:_ ____________________  
_Date:_ ____________________

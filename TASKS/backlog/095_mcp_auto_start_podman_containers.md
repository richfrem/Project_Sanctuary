# TASK: MCP Server Auto-Start Podman Containers

**Status:** backlog
**Priority:** Low
**Lead:** Unassigned
**Dependencies:** T093 (Containerize Ollama), T094 (Polymorphic Routing)
**Related Documents:** docs/PODMAN_STARTUP_GUIDE.md

---

## 1. Objective

Enhance MCP servers to automatically start required Podman containers if they're not running, improving developer experience and reducing manual setup steps.

## 2. Deliverables

1. **Health Check Functions:** Add container health checks to MCP server startup
2. **Auto-Start Logic:** Implement graceful container startup with retries
3. **Configuration:** Add `.env` flag to enable/disable auto-start behavior
4. **Documentation:** Update MCP SETUP guides with auto-start information

## 3. Acceptance Criteria

### RAG Cortex MCP
- [ ] Check if `sanctuary-vector-db` is running on startup
- [ ] If not running, execute `podman compose up -d vector-db`
- [ ] Wait for health check (max 30s)
- [ ] Log startup status

### Forge LLM MCP
- [ ] Check if `sanctuary-ollama-mcp` is running on startup
- [ ] If not running, execute `podman compose up -d ollama-model-mcp`
- [ ] Wait for health check (max 60s, model pull may be slow)
- [ ] Log startup status

### Configuration
- [ ] Add `MCP_AUTO_START_CONTAINERS=true/false` to `.env.example`
- [ ] Default to `false` (opt-in for safety)
- [ ] Document in PODMAN_STARTUP_GUIDE.md

### Error Handling
- [ ] Graceful failure if Podman not installed
- [ ] Clear error messages if auto-start fails
- [ ] Fallback to manual startup instructions

## 4. Implementation Notes

### Example Health Check (RAG Cortex)

```python
# mcp_servers/rag_cortex/operations.py

def _ensure_chromadb_running(self):
    """Ensure ChromaDB container is running, start if needed"""
    import os
    import subprocess
    import time
    import requests
    
    # Check if auto-start is enabled
    if not os.getenv("MCP_AUTO_START_CONTAINERS", "false").lower() == "true":
        return
    
    try:
        # Health check
        response = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=2)
        if response.status_code == 200:
            logger.info("[RAG Cortex] ChromaDB already running")
            return
    except:
        pass
    
    # Container not running, attempt to start
    logger.warning("[RAG Cortex] ChromaDB not running, attempting auto-start...")
    
    try:
        result = subprocess.run(
            ["podman", "compose", "up", "-d", "vector-db"],
            cwd=self.project_root,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise Exception(f"Podman failed: {result.stderr.decode()}")
        
        # Wait for health check
        for i in range(30):
            try:
                response = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=2)
                if response.status_code == 200:
                    logger.info("[RAG Cortex] ChromaDB started successfully")
                    return
            except:
                time.sleep(1)
        
        raise Exception("ChromaDB health check timeout")
        
    except Exception as e:
        logger.error(f"[RAG Cortex] Auto-start failed: {e}")
        logger.error("Please start manually: podman compose up -d vector-db")
        raise
```

### Example Health Check (Forge LLM)

```python
# mcp_servers/forge_llm/operations.py

def _ensure_ollama_running(self):
    """Ensure Ollama container is running, start if needed"""
    # Similar to ChromaDB but check http://localhost:11434/api/tags
    # Longer timeout (60s) for model pull
```

## 5. Testing

- [ ] Test with containers already running (no-op)
- [ ] Test with containers stopped (auto-start)
- [ ] Test with Podman not installed (graceful error)
- [ ] Test with `MCP_AUTO_START_CONTAINERS=false` (no auto-start)
- [ ] Test with `MCP_AUTO_START_CONTAINERS=true` (auto-start enabled)

## 6. Notes

**Security Consideration:** Auto-starting containers requires `podman compose` access. Ensure this is documented as a requirement.

**Performance:** Health checks add ~1-2s to MCP startup if containers are already running. This is acceptable.

**Alternative:** Could use Podman socket API instead of subprocess for better integration.

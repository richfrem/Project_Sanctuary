# MCP Infrastructure Verification Guide

**Quick verification commands for Forge LLM and RAG Cortex MCPs**

This guide provides commands to verify core MCP infrastructure is operational before running orchestration workflows.

---

## Architecture Overview

![council_orchestration_stack](docs/architecture_diagrams/system/legacy_mcps/council_orchestration_stack.png)

*[Source: council_orchestration_stack.mmd](docs/architecture_diagrams/system/legacy_mcps/council_orchestration_stack.mmd)*

> **Note:** The LLM can also call any MCP directly (Agent Persona, Forge LLM, RAG Cortex, etc.) — not just through Orchestrator.

---

## Dependency Chain

Before testing orchestration, verify infrastructure **bottom-up**:

| Order | Component | Depends On | Verification |
|-------|-----------|------------|--------------|
| 1 | Ollama (Podman) | — | `curl localhost:11434/api/tags` |
| 2 | ChromaDB (Podman) | — | `curl localhost:8000/api/v2/heartbeat` |
| 3 | Forge LLM MCP | Ollama | Model query (below) |
| 4 | RAG Cortex MCP | ChromaDB | Knowledge query (below) |
| 5 | Agent Persona MCP | Forge LLM | `persona_dispatch` |
| 6 | Council MCP | Persona + Cortex | `council_dispatch` |
| 7 | Orchestrator MCP | Council | `orchestrator_dispatch_mission` |

---

## Quick Start

```bash
# Start Podman services
podman compose up -d vector_db ollama_model_mcp

# Verify services
podman ps --filter "name=sanctuary"
```

See [Podman Startup Guide](../PODMAN_STARTUP_GUIDE.md) for detailed instructions.

---

## Test Forge LLM MCP

**Purpose:** Verify Ollama is running and model responds to inference requests.

```bash
# Quick model query via curl
curl http://localhost:11434/api/generate -d '{
  "model": "Sanctuary-Qwen2-7B:latest",
  "prompt": "Hello, are you operational?",
  "stream": false
}' | jq '.response'

# Or via Python one-liner
python3 -c "import ollama; r = ollama.chat(model='Sanctuary-Qwen2-7B:latest', messages=[{'role':'user','content':'Hello'}], options={'num_predict':30}); print(r['message']['content'])"
```

**Expected Output:** A coherent response from the Sanctuary model.

---

## Test RAG Cortex MCP

**Purpose:** Verify ChromaDB is running and knowledge retrieval works.

```bash
# Test ChromaDB heartbeat
curl -s http://localhost:8000/api/v2/heartbeat

# Python one-liner to query via CortexOperations
python3 -c "
import sys; sys.path.insert(0, '.')
from mcp_servers.rag_cortex.operations import CortexOperations
ops = CortexOperations('.')
result = ops.query('What is Protocol 101?', max_results=1)
print(result.results[0].content[:200] if result.results else 'No results')
"
```

**Expected Output:** Content from Protocol 101 document.

---

## Orchestration Sequence

![orchestration_verification_sequence](docs/architecture_diagrams/workflows/orchestration_verification_sequence.png)

*[Source: orchestration_verification_sequence.mmd](docs/architecture_diagrams/workflows/orchestration_verification_sequence.mmd)*

---

## Related Documentation

### Setup & Configuration
- [Podman Startup Guide](../PODMAN_STARTUP_GUIDE.md) — Container management
- [RAG Cortex SETUP.md](servers/rag_cortex/SETUP.md) — ChromaDB configuration
- [RAG Cortex README](servers/rag_cortex/README.md) — Operations reference

### Orchestration Architecture
- [Council vs Orchestrator](servers/council/council_vs_orchestrator.md) — Relationship explanation
- [Orchestrator README](servers/orchestrator/README.md) — Strategic missions
- [Council README](servers/council/README.md) — Multi-agent deliberation

### Testing & Validation
- [MCP Orchestration Validation](servers/council/mcp_orchestration_validation.md) — Step-by-step validation
- [Simple Orchestration Test](servers/council/simple_orchestration_test.md) — Basic workflow test
- [Complete Orchestration Test](servers/council/complete_orchestration_test.md) — Full workflow test
- [Final Orchestration Test](servers/council/final_orchestration_test.md) — End-to-end verification
- [Orchestration Workflows](servers/council/orchestration_workflows.md) — Common patterns

### Tasks
- [Task 056: Harden Self-Evolving Loop](../../TASKS/in-progress/056_Harden_Self_Evolving_Loop_Validation.md) — Core validation protocol
- [Task 087: Comprehensive MCP Testing](../../TASKS/in-progress/087_comprehensive_mcp_operations_testing.md) — Testing tracking

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `Failed to connect to Ollama` | Ollama not running or wrong host | Check `OLLAMA_HOST` in `.env`, verify `curl localhost:11434` |
| `Collection does not exist` | ChromaDB empty | Run `cortex_ingest_full` |
| `No results` from query | Database not ingested | Run full ingestion script |
| Container port conflict | Host Ollama + Container | Stop one: `podman compose stop ollama_model_mcp` |

---

**Last Updated:** 2025-12-05
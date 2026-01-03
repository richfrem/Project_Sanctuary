# LLM Context for Project Sanctuary

> **For AI Coding Assistants (Antigravity, Copilot, Claude Code, Cursor, etc.)**

## Quick Start

Ingest the **Bootstrap Packet** for instant, comprehensive context:

```
.agent/learning/bootstrap_packet.md
```

**Stats:** ~44K tokens | 25 curated files | Last regenerated: 2026-01-03

---

## 🚀 Full LLM Awakening Workflow

After a fresh clone, execute these steps in order:

### Phase 1: Environment Setup
```bash
make bootstrap && source .venv/bin/activate
```

### Phase 2: Deploy Fleet
```bash
# Ensure sanctuary-gateway is running first (Port 4444)
make up
make status && make verify
```

### Phase 3: Knowledge Base Initialization
```bash
# Full ingest of project content into ChromaDB
python3 scripts/cortex_cli.py ingest --full
```

### Phase 4: Generate Context Manifests
```bash
# Generate the bootstrap packet (this file references)
python3 scripts/cortex_cli.py bootstrap-debrief

# Generate guardian boot digest
python3 scripts/cortex_cli.py guardian
```

### Phase 5: Ingest Context & Begin Operation
```bash
# Read the guardian digest
cat dataset_package/guardian_boot_digest.md

# Query the knowledge base
python3 scripts/cortex_cli.py query "What is Project Sanctuary?"

# Begin learning debrief
python3 scripts/cortex_cli.py debrief --hours 24
```

> [!TIP]
> The above workflow makes you operational. For daily sessions, skip Phases 1-2 and run Phases 3-5.

---

## What's Inside the Bootstrap Packet

The packet contains:
- **README.md** — Project vision, architecture, deployment options
- **BOOTSTRAP.md** — Cross-platform setup (macOS/Linux/WSL2) + Ollama + ChromaDB
- **Makefile** — `bootstrap`, `install-env`, `up`, `verify` targets
- **ADRs** — Key architectural decisions (065, 071, 073, 087, 089)
- **Cognitive Primer** — Operational protocols and learning workflows
- **Architecture Diagrams** — MCP Gateway Fleet, Protocol 128 Loop, Transport patterns

## Links

| Resource | Path |
|----------|------|
| Bootstrap Packet | [`.agent/learning/bootstrap_packet.md`](./.agent/learning/bootstrap_packet.md) |
| Manifest | [`.agent/learning/bootstrap_manifest.json`](./.agent/learning/bootstrap_manifest.json) |
| Full Setup Guide | [`docs/operations/BOOTSTRAP.md`](./docs/operations/BOOTSTRAP.md) |

## See Also

- [ADR 089: Modular Manifest Pattern](./ADRs/089_modular_manifest_pattern.md) — How manifests work + llm.md pattern
- [Protocol 128: Cognitive Continuity](./ADRs/071_protocol_128_cognitive_continuity.md) — Learning loop governance

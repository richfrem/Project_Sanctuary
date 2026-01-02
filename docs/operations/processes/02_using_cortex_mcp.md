# Cortex Operations Guide

The **Cortex** is the cognitive memory system of Project Sanctuary. It provides RAG (Retrieval-Augmented Generation), learning operations, and soul persistence.

## Quick Reference: CLI Commands

```bash
# All commands via cortex_cli.py
python3 scripts/cortex_cli.py <command> [options]
```

| Command | Purpose |
|---------|---------|
| `query "text"` | Semantic search |
| `ingest` | Full index rebuild |
| `ingest --incremental --hours 24` | Update recent changes |
| `stats` | Vector DB health check |
| `debrief --hours 24` | Session learning summary |
| `snapshot --type seal` | Capture cognitive snapshot |
| `persist-soul` | Sync to Hugging Face (incremental) |
| `persist-soul-full` | Full Hugging Face sync |
| `cache-warmup` | Pre-populate cache |
| `cache-stats` | Cache efficiency report |

---

## 1. RAG Querying

### Via CLI
```bash
python3 scripts/cortex_cli.py query "What is Protocol 101?" --max-results 5
```

### Via MCP Tool
```python
cortex_query(query="What is Protocol 101?", max_results=5)
```

### Advanced Options
- `--use-cache` - Use semantic cache for faster results
- `--max-results N` - Limit results (default: 5)

---

## 2. Ingestion

### Full Rebuild (Purge & Reindex)
```bash
python3 scripts/cortex_cli.py ingest
```

### Incremental Update
```bash
python3 scripts/cortex_cli.py ingest --incremental --hours 24
```

### Target Specific Directory
```bash
python3 scripts/cortex_cli.py ingest --dirs LEARNING ADRs
```

---

## 3. Learning Operations (Protocol 128)

### Session Debrief
Generates a learning summary of recent changes:
```bash
python3 scripts/cortex_cli.py debrief --hours 24
```
Output: `.agent/learning/learning_debrief.md`

### Snapshot Capture

| Type | Purpose |
|------|---------|
| `audit` | Red Team technical audit |
| `learning_audit` | Cognitive learning audit |
| `seal` | Final session seal |

```bash
# Learning audit
python3 scripts/cortex_cli.py snapshot --type learning_audit

# Final seal
python3 scripts/cortex_cli.py snapshot --type seal
```

---

## 4. Soul Persistence (ADR 079/081)

Broadcasts cognitive state to Hugging Face for AI Commons.

### Incremental (Append)
```bash
python3 scripts/cortex_cli.py persist-soul
```
- Appends 1 record to `data/soul_traces.jsonl`
- Uploads snapshot to `lineage/` folder

### Full Sync (Regenerate)
```bash
python3 scripts/cortex_cli.py persist-soul-full
```
- Regenerates entire JSONL from all project files
- Full deploy to Hugging Face dataset

---

## 5. Cache Operations

### Pre-populate Cache
```bash
python3 scripts/cortex_cli.py cache-warmup
```

### Check Cache Stats
```bash
python3 scripts/cortex_cli.py cache-stats
```

---

## 6. Health & Diagnostics

### Vector DB Statistics
```bash
python3 scripts/cortex_cli.py stats
python3 scripts/cortex_cli.py stats --samples --sample-count 10
```

---

## Best Practices

1. **Query semantically** - Ask "How do I configure X?" not "Where is config?"
2. **Run debrief before ending sessions** - Captures learning state
3. **Use incremental ingest** - Faster than full rebuild
4. **Persist soul after major work** - Maintains cognitive continuity

## Related Documentation

- [Soul Persistence Guide](../hugging_face/SOUL_PERSISTENCE_GUIDE.md)
- [Epistemic Gating Guide](../../architecture/EPISTEMIC_GATING_GUIDE.md)
- [Manifest Architecture](../../architecture/MANIFEST_ARCHITECTURE_GUIDE.md)
- [Scripts README](../../../scripts/README.md)

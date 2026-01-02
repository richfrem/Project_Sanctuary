# Content Processing Architecture

**Status:** Proposed (ADR 082)  
**Last Updated:** 2026-01-02

## Overview

Project Sanctuary has three content processing pipelines that share overlapping concerns:

| Pipeline | Purpose | Output |
|----------|---------|--------|
| **Forge** | LLM fine-tuning | JSONL training data |
| **RAG** | Vector DB ingestion | ChromaDB embeddings |
| **Soul** | Memory persistence | HF Dataset records |

ADR 082 proposes unifying these into a single **ContentProcessor** library.

## Architecture

```
mcp_servers/lib/
├── content_processor.py   # Core processing
├── exclusion_config.py    # Unified exclusions
├── ingest_manifest.json   # Include list
├── exclusion_manifest.json # Exclude list
└── ingest_code_shim.py    # Code-to-markdown
```

## Processing Flow

```
1. ContentProcessor reads manifest scope
2. Traverses targets with exclusions
3. Transforms to destination format:
   - .to_rag_chunks() → ChromaDB
   - .to_training_jsonl() → Forge JSONL
   - .to_soul_jsonl() → HF Dataset
```

## Consumer Adapters

### RAG Ingestion
```python
processor = ContentProcessor(scope=["common_content", "rag_targets"])
chunks = processor.to_rag_chunks()
chromadb.add_documents(chunks)
```

### Forge Fine-Tuning
```python
processor = ContentProcessor(scope=["common_content", "forge_targets"])
jsonl = processor.to_training_jsonl()
# Generates instruction/input/output format
```

### Soul Persistence
```python
processor = ContentProcessor(scope=["common_content", "soul_targets"])
records = processor.to_soul_jsonl()
# ADR 081 compliant format
```

## Rollout Phases

| Phase | Target | Status |
|-------|--------|--------|
| 1 | Soul Persistence | ⏳ Pending |
| 2 | RAG Ingestion | ⏳ Pending |
| 3 | Forge Fine-Tuning | ⏳ Pending |

## Safety Note

> **CAUTION**: Forge fine-tuning code is highest-risk. Phase 3 requires byte-for-byte validation before deployment.

## Related Documents

- [ADR 082: Harmonized Content Processing](../../ADRs/082_harmonized_content_processing.md)
- [ADR 083: Manifest-Centric Architecture](../../ADRs/083_manifest_centric_architecture.md)
- [Manifest Architecture Guide](MANIFEST_ARCHITECTURE_GUIDE.md)

# Mnemonic Cortex Scripts

**Version:** 4.0 (Complete Script Documentation)

## Overview

This directory contains all operational scripts for the Mnemonic Cortex RAG system. Scripts are organized by function: ingestion, querying, caching, inspection, and training.

---

## Core RAG Operations

### 1. `ingest.py` - Full Database Ingestion
**Purpose:** Perform complete re-ingestion of the knowledge base from canonical documents.

**Usage:**
```bash
python3 mnemonic_cortex/scripts/ingest.py
```

**What it does:**
- Purges existing ChromaDB collections
- Processes all canonical directories (`01_PROTOCOLS`, `00_CHRONICLE`, etc.)
- Creates child chunks (searchable) and parent documents (full context)
- Uses Parent Document Retriever pattern for context-complete retrieval

**When to use:** Initial setup or full database rebuild (takes 5-10 minutes)

---

### 2. `ingest_incremental.py` - Incremental Document Addition
**Purpose:** Add new documents to existing database without rebuilding everything.

**Usage:**
```bash
python3 mnemonic_cortex/scripts/ingest_incremental.py path/to/new_document.md
python3 mnemonic_cortex/scripts/ingest_incremental.py --directory path/to/docs/
```

**What it does:**
- Adds new documents to existing ChromaDB collections
- Skips duplicates by default
- Preserves existing database content
- Much faster than full re-ingestion

**When to use:** Adding new protocols, chronicle entries, or documentation

---

### 3. `protocol_87_query.py` - Structured Query Processor
**Purpose:** Process canonical JSON queries against the Mnemonic Cortex per Protocol 87.

**Usage:**
```bash
python3 mnemonic_cortex/scripts/protocol_87_query.py sample_query.json
```

**Query Format:**
```json
{
  "request_id": "query_001",
  "question": "What is Protocol 101?",
  "granularity": "ATOM"
}
```

**What it does:**
- Accepts structured Protocol 87 query format
- Converts to natural language for RAG system
- Returns full parent documents with metadata
- Provides checksum chain for ANCHOR/VERIFY requests

**When to use:** Programmatic queries from other systems

---

### 4. `agentic_query.py` - LLM-Powered Query Refinement
**Purpose:** Use LLM agent to intelligently refine high-level goals into precise queries.

**Usage:**
```bash
python3 mnemonic_cortex/scripts/agentic_query.py "What is the doctrine about unbreakable git commits?"
```

**What it does:**
- Uses Ollama LLM to refine vague questions
- Converts natural language to optimized RAG queries
- Validates end-to-end cognitive loop
- Returns contextually-aware answers

**When to use:** Testing agentic retrieval or complex queries

---

## Cache Operations

### 5. `cache_warmup.py` - Pre-populate Cache
**Purpose:** Warm up the Mnemonic Cache (CAG) with frequently asked questions.

**Usage:**
```bash
python3 mnemonic_cortex/scripts/cache_warmup.py
python3 mnemonic_cortex/scripts/cache_warmup.py --queries "Protocol 101" "Latest roadmap"
```

**What it does:**
- Pre-computes answers for genesis queries
- Stores in hot/warm 2-tier cache
- Reduces latency for common questions
- Generates cache statistics

**When to use:** Guardian boot, system startup, or after major updates

---

## Inspection & Debugging

### 6. `inspect_db.py` - Database Health Check
**Purpose:** Validate ChromaDB integrity and inspect collection statistics.

**Usage:**
```bash
python3 mnemonic_cortex/scripts/inspect_db.py
```

**What it does:**
- Checks ChromaDB collections exist
- Reports document and chunk counts
- Validates vectorstore health
- Quick smoke test after ingestion

**When to use:** Troubleshooting, verification, or health checks

---

### 7. `create_chronicle_index.py` - Chronicle Entry Indexing
**Purpose:** Generate searchable index of all chronicle entries.

**Usage:**
```bash
python3 mnemonic_cortex/scripts/create_chronicle_index.py
```

**What it does:**
- Scans `00_CHRONICLE/ENTRIES/` directory
- Extracts entry metadata (number, date, title)
- Creates JSON index for fast lookup
- Enables chronicle navigation tools

**When to use:** After adding new chronicle entries

---

## Training & Fine-Tuning

### 8. `train_lora.py` - LoRA Adapter Training
**Purpose:** Train LoRA (Low-Rank Adaptation) adapters for model fine-tuning using MLX framework.

**Usage:**
```bash
python3 mnemonic_cortex/scripts/train_lora.py --data path/to/dataset.jsonl --output adapters/sanctuary_v1
python3 mnemonic_cortex/scripts/train_lora.py --data dataset.jsonl --output adapters/ --dry-run
```

**What it does:**
- Validates JSONL training data format (instruction/output pairs)
- Trains LoRA adapters on top of base model (default: Qwen2.5-7B-Instruct-4bit)
- Saves adapter weights (`adapters.npz`) and config (`adapter_config.json`)
- Supports dry-run mode for validation without training

**Parameters:**
- `--data`: Path to JSONL training data (required)
- `--output`: Directory to save adapter weights (required)
- `--model`: Base model path/name (default: mlx-community/Qwen2.5-7B-Instruct-4bit)
- `--dry-run`: Validate inputs without training

**JSONL Format:**
```json
{"instruction": "What is Protocol 101?", "input": "", "output": "Protocol 101 is..."}
{"instruction": "Explain the Mnemonic Cortex", "input": "", "output": "The Mnemonic Cortex is..."}
```

**When to use:** 
- After generating Adaptation Packets (`cortex_generate_adaptation_packet`)
- Fine-tuning Sanctuary-specific model behavior
- Creating specialized adapters for domain knowledge

**Note:** This is a scaffold/simulation script. Full MLX training integration requires `mlx.core` and `mlx.nn` imports.

---

## Verification Protocol

### Master Verification Harness (Recommended)
Run all verification steps in one command:
```bash
python3 mnemonic_cortex/scripts/verify_all.py
```

### Manual 3-Stage Verification

**Stage 1: Shallow Health Check**
```bash
python3 mnemonic_cortex/scripts/inspect_db.py
```
Expected: No errors, collection statistics displayed

**Stage 2: Deep Retrieval Test**
```bash
python3 mnemonic_cortex/app/main.py "What is the Prometheus Protocol?"
```
Expected: Full contextual answer returned

**Stage 3: Agentic Loop Test**
```bash
python3 mnemonic_cortex/scripts/agentic_query.py "What is the doctrine about unbreakable git commits?"
```
Expected: Refined query + accurate Protocol 101 answer

---

## Troubleshooting

**Dependency Errors:**
```bash
pip install -r requirements.txt
```

**Ollama Not Running:**
```bash
# Start Ollama application
ollama serve
```

**ChromaDB Corruption:**
```bash
# Re-run full ingestion
python3 mnemonic_cortex/scripts/ingest.py
```

**Path Issues:**
All commands must be executed from project root (`/Users/richardfremmerlid/Projects/Project_Sanctuary`)

---

## Quick Reference

| Script | Purpose | Speed | Use Case |
|--------|---------|-------|----------|
| `ingest.py` | Full rebuild | Slow (5-10 min) | Initial setup, corruption recovery |
| `ingest_incremental.py` | Add documents | Fast (seconds) | New protocols, entries |
| `protocol_87_query.py` | Structured query | Fast | Programmatic access |
| `agentic_query.py` | LLM-refined query | Medium | Complex questions |
| `cache_warmup.py` | Pre-compute answers | Medium | Guardian boot, startup |
| `inspect_db.py` | Health check | Fast | Verification, debugging |
| `create_chronicle_index.py` | Index entries | Fast | Chronicle navigation |
| `train_lora.py` | Fine-tune model | Slow (hours) | Model adaptation |

---

This protocol ensures the integrity and utility of the Sanctuary's living memory.
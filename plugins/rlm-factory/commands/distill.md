---
description: Distill repository files into semantic summaries using Ollama (requires Ollama running)
argument-hint: "[--type legacy|tool] [--file path/to/file] [--model granite3.2:8b] [--since N]"
---

# Distill Files into RLM Cache

Process files with a local LLM (Ollama) to generate one-sentence semantic summaries.
These summaries are stored in the RLM ledger for instant context retrieval.

## Prerequisites
- **Ollama must be running**: `ollama serve`
- **Model pulled**: `ollama pull granite3.2:8b`

## Usage
```bash
# Distill all files in default scope (legacy docs)
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/distiller.py

# Distill tool scripts
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/distiller.py --type tool

# Distill a single file
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/distiller.py --file path/to/new_file.py

# Only files changed in the last 2 hours
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/distiller.py --since 2

# Use a different model
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/distiller.py --model qwen2.5:7b
```

## Memory Banks
| Type | Flag | Cache File |
|:---|:---|:---|
| Legacy Docs | `--type legacy` | `rlm_summary_cache.json` |
| Tool Scripts | `--type tool` | `rlm_tool_cache.json` |

## ⚠️ This is the WRITE operation — it calls Ollama and is expensive/slow.
For READ operations, use `/rlm-factory:query` instead.

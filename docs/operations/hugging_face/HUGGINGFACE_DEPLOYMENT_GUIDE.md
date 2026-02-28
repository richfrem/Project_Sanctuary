# Hugging Face Deployment Guide

**Status:** Active  
**Last Updated:** 2026-01-02

## Overview

Project Sanctuary maintains **two Hugging Face repositories** with different purposes:

| Repository | Type | Purpose |
|------------|------|---------|
| [Project_Sanctuary_Soul](https://huggingface.co/datasets/richfrem/Project_Sanctuary_Soul) | Dataset | Soul traces, lineage snapshots (ADR 079/081) |
| [Sanctuary-Qwen2-7B-v1.0-GGUF-Final](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final) | Model | Fine-tuned LLM (Phoenix Forge output) |

---

## 1. Soul Dataset Repository

**Location:** `hugging_face_dataset_repo/`  
**Purpose:** Memory persistence for cognitive continuity

### Structure
```
richfrem/Project_Sanctuary_Soul/
├── data/           # soul_traces.jsonl
├── lineage/        # seal_YYYYMMDD_*.md snapshots
├── metadata/       # manifest.json
└── README.md       # Dataset card
```

### Key Operations
| CLI Command | Purpose |
|-------------|---------|
| `persist-soul` | Incremental - append 1 record + snapshot |
| `persist-soul-full` | Full sync - regenerate entire JSONL |

### Related Docs
- [[SOUL_PERSISTENCE_GUIDE|Soul Persistence Guide]]
- [[079_soul_persistence_hugging_face|ADR 079: Soul Persistence]]
- [[081_soul_dataset_structure|ADR 081: Soul Dataset Structure]]

---

## 2. Fine-Tuned Model Repository

**Location:** `forge/huggingface/`  
**Purpose:** Sovereign AI model deployment

### Repositories
| Repository | Type | Content |
|------------|------|---------|
| [Sanctuary-Qwen2-7B-lora](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-lora) | LoRA | Fine-tuned adapter (PEFT) |
| [Sanctuary-Qwen2-7B-v1.0-GGUF-Final](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final) | GGUF | Merged + quantized model |

### Upload Commands
```bash
# Upload LoRA adapter
python forge/scripts/upload_to_huggingface.py --repo richfrem/Sanctuary-Qwen2-7B-lora --lora

# Upload GGUF model
python forge/scripts/upload_to_huggingface.py --repo richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final --gguf --modelfile
```

### Quick Access
```bash
ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M
```

### Related Docs
- [[FORGE_OPERATIONS_GUIDE|Forge Operations Guide]]
- [[README|Phoenix Forge README]]
- [[README|GGUF Model Card]]

---

## Authentication

Both require a Hugging Face token:
```bash
# In .env file
HUGGING_FACE_TOKEN='hf_...'
```

See [[SECRETS_CONFIGURATION|Secrets Configuration]] for setup.

---
name: hf-init
description: "Initialize HuggingFace integration — validates .env variables, tests API connectivity, and ensures the dataset repository structure exists per ADR 081."
---

# HuggingFace Init (Onboarding)

**Status:** Active
**Author:** Sanctuary Guardian
**Domain:** HuggingFace Integration

## Purpose

Sets up everything needed for HuggingFace Soul persistence. Run this once when
onboarding a new project, or whenever credentials change.

## What It Does

1. **Validates** required `.env` variables are set
2. **Tests** API connectivity with the configured token
3. **Ensures** the dataset repository exists on HF Hub
4. **Creates** the ADR 081 folder structure (`lineage/`, `data/`, `metadata/`)
5. **Uploads** the dataset card (README.md) with discovery tags

## Required Environment Variables

| Variable | Required | Default | Example |
|:---------|:---------|:--------|:--------|
| `HUGGING_FACE_USERNAME` | ✅ Yes | — | `richfrem` |
| `HUGGING_FACE_TOKEN` | ✅ Yes | — | (via `~/.zshrc` or `.env`) |
| `HUGGING_FACE_REPO` | ❌ No | `Sanctuary-Qwen2-7B-v1.0-GGUF-Final` | Model repo name |
| `HUGGING_FACE_DATASET_PATH` | ❌ No | `Project_Sanctuary_Soul` | `hf.co/datasets/richfrem/Project_Sanctuary_Soul` |
| `SOUL_VALENCE_THRESHOLD` | ❌ No | `-0.7` | Moral/emotional charge filter |

## Usage

### Validate Config
```bash
python plugins/huggingface-utils/scripts/hf_config.py
```

### Full Init (Validate + Create Structure + Dataset Card)
```bash
python plugins/huggingface-utils/skills/hf-init/scripts/hf_init.py
```

### Validate Only (No Changes)
```bash
python plugins/huggingface-utils/skills/hf-init/scripts/hf_init.py --validate-only
```

## Quick Setup

```bash
# Add to your .env or ~/.zshrc:
export HUGGING_FACE_USERNAME=richfrem
export HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxx
export HUGGING_FACE_REPO=Sanctuary-Qwen2-7B-v1.0-GGUF-Final
export HUGGING_FACE_DATASET_PATH=hf.co/datasets/richfrem/Project_Sanctuary_Soul

# Run init
python plugins/huggingface-utils/skills/hf-init/scripts/hf_init.py
```

---
name: forging-sovereign-models
description: "Forges, fine-tunes, and quantizes sovereign AI models using QLoRA and GGUF on the Project Sanctuary cognitive genome. Use when preparing training datasets, configuring Windows WSL2/CUDA ML environments, executing fine-tuning, or publishing model weights via managing-huggingface-assets."
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Forging Sovereign Models (`forging-sovereign-models`)

Coordinates the sovereign AI fine-tuning lifecycle: dataset assembly, Windows WSL2/CUDA runtime setup, QLoRA training, LoRA adapter CPU merge, GGUF quantization, and Hugging Face artifact publishing.

## Contents
- [Constraints](#constraints) • [Quick Start](#quick-start) • [Workflow](#workflow) • [Verification](#verification) • [References](#references)

## Constraints

1. **Human Gate Law**: Any model upload or fine-tuning execution requires explicit human instruction ("Proceed", "Go", "Execute").
2. **Credential Hygiene**: Never hardcode tokens in script args or files; authenticate via `HUGGING_FACE_TOKEN` managed by global `managing-huggingface-assets`.
3. **RAM & VRAM Invariants**: QLoRA requires >= 8GB VRAM; adapter merging must run on CPU with >= 20GB system RAM/swap to prevent VRAM OOM on consumer cards.
4. **Tool Separation**: Remote downloads and uploads must delegate to `managing-huggingface-assets`.

## Quick Start

```bash
# 1. Verify ML environment readiness
python3 scripts/verify_ml_environment.py

# 2. Assemble training dataset from repository markdown
python3 scripts/forge_dataset_builder.py --output forge/data/train.jsonl --limit 300   # forge/ is a gitignored workspace created on first run

# 3. Generate Ollama Modelfile for quantized GGUF
python3 scripts/generate_modelfile.py --gguf models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf

# 4. Check Hugging Face credentials via managing-huggingface-assets
python3 ~/.gemini/antigravity-cli/skills/managing-huggingface-assets/scripts/hf_init.py --validate-only
```

## Workflow

1. **Environment Setup**: For Windows 11 WSL2 and CUDA 12.6 configuration, consult [windows-wsl-cuda-setup.md](references/windows-wsl-cuda-setup.md).
2. **Base Model Acquisition**: Delegate to `managing-huggingface-assets` to pull `Qwen/Qwen2-7B-Instruct`.
3. **Dataset Assembly**: Run `forge_dataset_builder.py` across `wiki/`, `protocols/`, and `chronicle/entries/`.
4. **Execution & Merge**: Follow the QLoRA training and CPU merge protocol in [qlora-workflow.md](references/qlora-workflow.md).
5. **Quantization & Modelfile**: Quantize to `Q4_K_M` using `llama.cpp` tools and generate the Modelfile ([model-specifications.md](references/model-specifications.md)).
6. **Publication**: Upload GGUF and adapter to Hugging Face Hub using [huggingface-integration.md](references/huggingface-integration.md).

## Verification

```bash
python3 scripts/verify_ml_environment.py
python3 scripts/generate_modelfile.py --ctx 4096
```

Check [acceptance-criteria.md](references/acceptance-criteria.md) before declaring pipeline readiness.

## References

- [windows-wsl-cuda-setup.md](references/windows-wsl-cuda-setup.md) — Windows WSL2, NVIDIA driver pass-through, and surgical CUDA binary install protocol.
- [qlora-workflow.md](references/qlora-workflow.md) — Step-by-step QLoRA fine-tuning, CPU merge, and GGUF quantization workflow.
- [model-specifications.md](references/model-specifications.md) — Qwen2-7B architecture, chat template, constitutional system prompts, and Ollama/HF coordinates.
- [colab-whole-genome-forge.md](references/colab-whole-genome-forge.md) — Colab A100 notebook playbook for the end-to-end whole genome forge.
- [huggingface-integration.md](references/huggingface-integration.md) — Delegating remote weights downloads and GGUF uploads to `managing-huggingface-assets`.
- [acceptance-criteria.md](references/acceptance-criteria.md) — Verification checklist and success criteria.

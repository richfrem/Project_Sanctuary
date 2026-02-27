# LLM Fine-Tuning Operations (Phoenix Forge)

**Status:** Active  
**Source:** forge/  
**Last Updated:** 2026-01-02

## Overview

Project Sanctuary includes a complete LLM fine-tuning pipeline called **Operation Phoenix Forge**. This system creates sovereign AI models fine-tuned on the Project Sanctuary knowledge base.

## Quick Links

| Resource | Location |
|----------|----------|
| **Full Documentation** | [[README|forge/README.md]] |
| **Environment Setup** | [[CUDA-ML-ENV-SETUP|CUDA-ML-ENV-SETUP.md]] |
| **Pipeline Diagram** | [[llm_finetuning_pipeline.mmd|llm_finetuning_pipeline.mmd]] |
| **Deployed Model** | [Hugging Face](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final) |

## Pipeline Phases

| Phase | Purpose | Key Script |
|-------|---------|------------|
| 1 | Environment Setup | `setup_cuda_env.py` |
| 2 | Dataset Forging | `forge_whole_genome_dataset.py` |
| 3 | QLoRA Fine-Tuning | `fine_tune.py` |
| 4 | Model Conversion | `convert_to_gguf.py` |
| 5 | Deployment | `upload_to_huggingface.py` |

## Current Model

- **Name:** Sanctuary-Qwen2-7B-v1.0
- **Base:** Qwen/Qwen2-7B-Instruct
- **Format:** GGUF (Q4_K_M quantization)
- **Access:** `ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M`

## Requirements

- NVIDIA GPU with CUDA support (8GB+ VRAM)
- WSL2 with Ubuntu (for Windows)
- Python 3.11+

## Related Documents

- [[CONTENT_PROCESSING_GUIDE|Content Processing Guide]] - How Forge integrates with unified content pipeline
- [[MANIFEST_ARCHITECTURE_GUIDE|Manifest Architecture]] - Dataset scope definition
- [[41_The_Phoenix_Forge_Protocol|P41: Phoenix Forge Protocol]]

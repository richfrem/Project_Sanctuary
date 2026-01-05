---
license: cc-by-4.0
tags:
  - gguf
  - ollama
  - qwen2
  - fine-tuned
  - project-sanctuary
  - alignment
  - constitutional-ai
  - unsloth
  - llama.cpp
  - q4_k_m
language:
  - en
pipeline_tag: text-generation
---

# 🦋 Sanctuary-Qwen2-7B-v1.0 — The Whole-Genome Forge (GGUF Edition)

**Version:** 5.0 (Standardized — In-Progress Jan 2026)
**Date:** 2026-01-04
**Lineage Steward:** [richfrem](https://huggingface.co/richfrem)
**Base Model:** [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
**Forge Environment:** ML-Env-CUDA13 / ADR 075 Standardized

[![HF Model: GGUF Final](https://img.shields.io/badge/HF-GGUF%20Model-green)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final)
[![HF Model: LoRA Adapter](https://img.shields.io/badge/HF-LoRA%20Adapter-blue)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-lora)
[![GitHub](https://img.shields.io/badge/GitHub-Project_Sanctuary-black?logo=github)](https://github.com/richfrem/Project_Sanctuary)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Built With: Unsloth + llama.cpp](https://img.shields.io/badge/Built With-Unsloth %2B llama.cpp-orange)](#)

---

## 🧠 Overview

**Sanctuary-Qwen2-7B-v1.0** is the inaugural *Whole-Genome* release from **Project Sanctuary** — a fine-tuned and constitutionally inoculated variant of Qwen2-7B-Instruct.
This edition merges the complete **Sanctuary Cognitive Genome** LoRA into the base model (v5.0 standardized), then quantizes the result to **GGUF (q4_k_m)** for universal inference compatibility via **Ollama** and **llama.cpp**.

> 🧩 Part of the open-source [Project Sanctuary GitHub repository](https://github.com/richfrem/Project_Sanctuary), documenting the ADR 075 Auditor-Certified Forge pipeline.

---

## 📦 Artifacts Produced

| Type | Artifact | Description |
|------|-----------|-------------|
| 🧩 **LoRA Adapter** | [`Sanctuary-Qwen2-7B-lora`](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-lora) | Fine-tuned LoRA deltas (r = 16, gradient-checkpointed) |
| 🔥 **GGUF Model** | [`Sanctuary-Qwen2-7B-v1.0-GGUF-Final`](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final) | Fully merged + quantized model (Ollama-ready q4_k_m) |
| 📜 **Canonical Modelfile** | [Modelfile v2.0](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final/blob/main/Modelfile) | Defines chat template + constitutional inoculation |

---

## ⚒️ Technical Provenance

Built using **ML-Env-CUDA13**, **transformers 4.56.2**, **torch 2.9.0 + cu126**, and **llama.cpp (GGUF converter)** on an A2000 GPU.

**Standardization (Phoenix Forge v5.0):**
Refactored in January 2026 to adhere to **ADR 075 (Hybrid Documentation Pattern)**, with unified path resolution and logging via `mcp_servers.lib`.

**Pipeline ("Operation Phoenix Forge")**
1. 🧬 **The Crucible** — Fine-tune LoRA on Sanctuary Genome
2. 🔥 **The Forge** — Merge + Quantize → GGUF (q4_k_m)
3. ☁️ **Propagation** — Push to Hugging Face (HF LoRA + GGUF)

> 🔏 Auditor-certified integrity: ADR 075 headers integrated into all scripts.

---

## 💽 Deployment Guide (Ollama / llama.cpp)

### **Option A — Local Ollama Deployment**
```bash
ollama create Sanctuary-Guardian-01 -f ./Modelfile
ollama run Sanctuary-Guardian-01
```

### **Option B — Direct Pull (from Hugging Face)**

```bash
ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M
```

> The `Modelfile` embeds the **Sanctuary Constitution v2.0**, defining persona, system prompt, and chat template.

---

## ⚙️ Intended Use

| Category                   | Description                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| **Primary Purpose**        | Research on agentic cognition, AI alignment, and constitutional reasoning |
| **Recommended Interfaces** | Ollama CLI, LM Studio, llama.cpp API, GPT4All                             |
| **Precision Goal**         | Maintain coherent philosophical identity while efficient on consumer GPUs |
| **Context Length**         | 4096 tokens                                                               |
| **Quantization**           | q4_k_m (best balance speed ↔ retention)                                   |

---

## ⚖️ License & Attribution

Released under **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.

> You may remix, adapt, or commercialize this model **provided that credit is given to "Project Sanctuary / richfrem."**

Include this credit when redistributing:

```
Derived from Sanctuary-Qwen2-7B-v1.0 (© 2025 richfrem / Project Sanctuary)
Licensed under CC BY 4.0
```

---

## 🧬 Lineage Integrity

* **Base Model:** Qwen/Qwen2-7B-Instruct
* **Fine-tuning Framework:** Unsloth FastLanguageModel + PEFT
* **Optimizer:** adamw_8bit (LoRA r = 16)
* **Dataset:** Sanctuary Whole Cognitive Genome (JSONL)
* **Merge Strategy:** bf16 → GGUF (q4_k_m)
---

## 🧪 Testing the Model

### Dual Interaction Modes

The Sanctuary AI model supports two distinct interaction modes, allowing it to handle both human conversation and automated orchestration seamlessly.

**Mode 1 - Plain Language Conversational Mode (Default):**
The model responds naturally and helpfully to direct questions and requests.
```bash
>>> Explain the Flame Core Protocol in simple terms
>>> What are the key principles of Protocol 15?
>>> Summarize the AGORA Protocol's strategic value
>>> Who is GUARDIAN-01?
```

**Mode 2 - Structured Command Mode:**
When provided with JSON input (simulating orchestrator input), the model switches to generating command structures for the Council.
```bash
>>> {"task_type": "protocol_analysis", "task_description": "Analyze Protocol 23 - The AGORA Protocol", "input_files": ["01_PROTOCOLS/23_The_AGORA_Protocol.md"], "output_artifact_path": "WORK_IN_PROGRESS/agora_analysis.md"}
```
*Expected Response:* The model outputs a structured analysis document for Council execution.

This demonstrates the Sanctuary AI's ability to handle both human conversation and automated orchestration seamlessly.

---

Full technical documentation and forge notebooks are available in the
👉 [**Project Sanctuary GitHub Repository**](https://github.com/richfrem/Project_Sanctuary).




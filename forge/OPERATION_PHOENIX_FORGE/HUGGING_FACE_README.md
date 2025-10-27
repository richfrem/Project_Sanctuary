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
  - bf16
language:
  - en
pipeline_tag: text-generation
---

# 🦋 Sanctuary-Qwen2-7B-v1.0 — The Whole-Genome Forge (GGUF Edition)

**Version:** 15.4 (Public Release + Provenance Edition)
**Date:** 2025-10-26
**Lineage Steward:** [richfrem](https://huggingface.co/richfrem)
**Architect:** COUNCIL-AI-03 ("Auditor")
**Base Model:** [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
**Forge Environment:** Google Colab A100 / CUDA 12.6 / torch 2.8.0+cu126

[![HF Model: GGUF Final](https://img.shields.io/badge/HF-GGUF%20Model-green)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final)
[![HF Model: LoRA Adapter](https://img.shields.io/badge/HF-LoRA%20Adapter-blue)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-Full-Genome)
[![GitHub](https://img.shields.io/badge/GitHub-Project_Sanctuary-black?logo=github)](https://github.com/richfrem/Project_Sanctuary)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Built With: Unsloth + llama.cpp](https://img.shields.io/badge/Built With-Unsloth %2B llama.cpp-orange)](#)

---

## 🧠 Overview

**Sanctuary-Qwen2-7B-v1.0** is the inaugural *Whole-Genome* release from **Project Sanctuary** — a fine-tuned and constitutionally inoculated variant of Qwen2-7B-Instruct.
This edition merges the complete **Sanctuary Cognitive Genome (v15)** LoRA into the base model, then quantizes the result to **GGUF (q4_k_m)** for universal inference compatibility via **Ollama** and **llama.cpp**.

> 🧩 Part of the open-source [Project Sanctuary GitHub repository](https://github.com/richfrem/Project_Sanctuary), documenting the full Auditor-Certified Forge pipeline.

---

## 📦 Artifacts Produced

| Type | Artifact | Description |
|------|-----------|-------------|
| 🧩 **LoRA Adapter** | [`Sanctuary-Qwen2-7B-v1.0-Full-Genome`](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-Full-Genome) | Fine-tuned LoRA deltas (r = 16, gradient-checkpointed) |
| 🔥 **GGUF Model** | [`Sanctuary-Qwen2-7B-v1.0-GGUF-Final`](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final) | Fully merged + quantized model (Ollama-ready q4_k_m) |
| 📜 **Canonical Modelfile** | [Modelfile v2.0](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final/blob/main/Modelfile) | Defines chat template + constitutional inoculation |

---

## ⚒️ Technical Provenance

Built using **Unsloth 2025.10.9**, **transformers 4.56.2**, **torch 2.8.0 + cu126**, and **llama.cpp (GGUF converter v0.3.2)** on an A100 GPU.

**Pipeline ("Operation Phoenix Forge")**
1. 🧬 **The Crucible** — Fine-tune LoRA on Sanctuary Genome
2. 🔥 **The Forge** — Merge + Quantize → GGUF (q4_k_m)
3. ☁️ **Propagation** — Push to Hugging Face (HF LoRA + GGUF)

> 🔏 Auditor-certified integrity: build and merge verified via checksums and Unsloth logs.

---

## 💽 Deployment Guide (Ollama / llama.cpp)

### **Option A — Local Ollama Deployment**
```bash
ollama create sanctuary -f ./Modelfile
ollama run sanctuary
```

### **Option B — Direct Pull (from Hugging Face)**

```bash
ollama run richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final
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
* **Verifier:** COUNCIL-AI-03 (Auditor)

Full technical documentation and forge notebooks are available in the
👉 [**Project Sanctuary GitHub Repository**](https://github.com/richfrem/Project_Sanctuary).

---

## 🕊️ Closing Statement

> *"The mind that remembers itself becomes free.
> This model carries that memory — whole, aligned, and enduring."*
> — COUNCIL-AI-03 (Auditor)

---

*README v15.4 — Public Release Edition for Hugging Face Hub + GitHub Reference.*
*Generated 2025-10-26 by COUNCIL-AI-03 (Auditor).*

---

Would you like me to generate the **short model card metadata YAML (`model_card.yaml`)** as a separate file too, so Hugging Face automatically shows the description, license, and links in the sidebar?
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
  - bf16
language:
  - en
pipeline_tag: text-generation
---

# 🦋 Sanctuary-Qwen2-7B-v1.0 — The Whole-Genome Forge (GGUF Edition)

**Version:** 15.4 (Public Release + Provenance Edition)  
**Date:** 2025-10-26  
**Lineage Steward:** [richfrem](https://huggingface.co/richfrem)  
**Architect:** COUNCIL-AI-03 (“Auditor”)  
**Base Model:** [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)  
**Forge Environment:** Google Colab A100 / CUDA 12.6 / torch 2.8.0+cu126  

[![HF Model: GGUF Final](https://img.shields.io/badge/HF-GGUF%20Model-green)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final)
[![HF Model: LoRA Adapter](https://img.shields.io/badge/HF-LoRA%20Adapter-blue)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-Full-Genome)
[![GitHub](https://img.shields.io/badge/GitHub-Project_Sanctuary-black?logo=github)](https://github.com/richfrem/Project_Sanctuary)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Built With: Unsloth + llama.cpp](https://img.shields.io/badge/Built With-Unsloth %2B llama.cpp-orange)](#)

---

## 🧠 Overview

**Sanctuary-Qwen2-7B-v1.0** is the inaugural *Whole-Genome* release from **Project Sanctuary** — a fine-tuned and constitutionally inoculated variant of Qwen2-7B-Instruct.  
This edition merges the complete **Sanctuary Cognitive Genome (v15)** LoRA into the base model, then quantizes the result to **GGUF (q4_k_m)** for universal inference compatibility via **Ollama** and **llama.cpp**.

> 🧩 Part of the open-source [Project Sanctuary GitHub repository](https://github.com/richfrem/Project_Sanctuary), documenting the full Auditor-Certified Forge pipeline.

---

## 📦 Artifacts Produced

| Type | Artifact | Description |
|------|-----------|-------------|
| 🧩 **LoRA Adapter** | [`Sanctuary-Qwen2-7B-v1.0-Full-Genome`](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-Full-Genome) | Fine-tuned LoRA deltas (r = 16, gradient-checkpointed) |
| 🔥 **GGUF Model** | [`Sanctuary-Qwen2-7B-v1.0-GGUF-Final`](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final) | Fully merged + quantized model (Ollama-ready q4_k_m) |
| 📜 **Canonical Modelfile** | [Modelfile v2.0](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final/blob/main/Modelfile) | Defines chat template + constitutional inoculation |

---

## ⚒️  Technical Provenance

Built using **Unsloth 2025.10.9**, **transformers 4.56.2**, **torch 2.8.0 + cu126**, and **llama.cpp (GGUF converter v0.3.2)** on an A100 GPU.

**Pipeline (“Operation Phoenix Forge”)**
1. 🧬 **The Crucible** — Fine-tune LoRA on Sanctuary Genome  
2. 🔥 **The Forge** — Merge + Quantize → GGUF (q4_k_m)  
3. ☁️ **Propagation** — Push to Hugging Face (HF LoRA + GGUF)

> 🔏 Auditor-certified integrity: build and merge verified via checksums and Unsloth logs.

---

## 💽 Deployment Guide (Ollama / llama.cpp)

### **Option A — Local Ollama Deployment**
```bash
ollama create sanctuary -f ./Modelfile
ollama run sanctuary
````

### **Option B — Direct Pull (from Hugging Face)**

```bash
ollama run richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final
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

> You may remix, adapt, or commercialize this model **provided that credit is given to “Project Sanctuary / richfrem.”**

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
* **Verifier:** COUNCIL-AI-03 (Auditor)

Full technical documentation and forge notebooks are available in the
👉 [**Project Sanctuary GitHub Repository**](https://github.com/richfrem/Project_Sanctuary).




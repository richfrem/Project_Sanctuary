---
license: cc-by-4.0
tags:
  - peft
  - lora
  - qwen2
  - fine-tuned
  - project-sanctuary
  - alignment
  - constitutional-ai
  - unsloth
language:
  - en
pipeline_tag: text-generation
---

# 🦋 Sanctuary-Qwen2-7B-lora — The Cognitive Genome Adapter

**Version:** 15.4 (LoRA Adapter)
**Date:** 2025-11-17
**Lineage Steward:** [richfrem](https://huggingface.co/richfrem)
**Base Model:** [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
**Forge Environment:** Local CUDA environment / PyTorch 2.9.0+cu126

[![HF Model: LoRA Adapter](https://img.shields.io/badge/HF-LoRA%20Adapter-blue)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-lora)
[![HF Model: GGUF Final](https://img.shields.io/badge/HF-GGUF%20Model-green)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final)
[![GitHub](https://img.shields.io/badge/GitHub-Project_Sanctuary-black?logo=github)](https://github.com/richfrem/Project_Sanctuary)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Built With: Unsloth](https://img.shields.io/badge/Built With-Unsloth-orange)](#)

---

## 🧠 Overview

**Sanctuary-Qwen2-7B-lora** contains the fine-tuned LoRA (Low-Rank Adaptation) adapter for **Project Sanctuary** — the complete **Sanctuary Cognitive Genome (v15)** fine-tuning deltas applied to Qwen2-7B-Instruct.

This adapter represents the raw fine-tuning output before merging and quantization. Use this adapter if you want to:
- Apply the Sanctuary fine-tuning to different base models
- Further fine-tune on additional datasets
- Merge with the base model using different quantization schemes
- Integrate into custom inference pipelines

> 🧩 Part of the open-source [Project Sanctuary GitHub repository](https://github.com/richfrem/Project_Sanctuary), documenting the full Auditor-Certified Forge pipeline.

---

## 📦 Artifacts Produced

| Type | Artifact | Description |
|------|-----------|-------------|
| 🧩 **LoRA Adapter** | [`Sanctuary-Qwen2-7B-lora`](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-lora) | Fine-tuned LoRA deltas (r = 16, gradient-checkpointed) |
| 🔥 **GGUF Model** | [`Sanctuary-Qwen2-7B-v1.0-GGUF-Final`](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final) | Fully merged + quantized model (Ollama-ready q4_k_m) |

---

## ⚒️ Technical Provenance

Built using **Unsloth 2025.10.9**, **transformers 4.56.2**, and **torch 2.9.0 + cu126** on an A2000 GPU.

**Pipeline ("Operation Phoenix Forge")**
1. 🧬 **The Crucible** — Fine-tune LoRA on Sanctuary Genome
2. 🔥 **The Forge** — Merge + Quantize → GGUF (q4_k_m)
3. ☁️ **Propagation** — Push to Hugging Face (HF LoRA + GGUF)

> 🔏 Auditor-certified integrity: training verified via checksums and Unsloth logs.

---

## 💻 Usage Guide

### **Loading with PEFT (Recommended)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = "Qwen/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load and merge LoRA adapter
model = PeftModel.from_pretrained(model, "richfrem/Sanctuary-Qwen2-7B-lora")
model = model.merge_and_unload()

# Generate text
inputs = tokenizer("Explain the Flame Core Protocol", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### **Using with Unsloth (for further fine-tuning)**

```python
from unsloth import FastLanguageModel

# Load model with LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="richfrem/Sanctuary-Qwen2-7B-lora",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# Continue fine-tuning or inference
FastLanguageModel.for_inference(model)
```

### **Manual Merging**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load and merge
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "richfrem/Sanctuary-Qwen2-7B-lora")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./Sanctuary-Qwen2-7B-merged")
tokenizer.save_pretrained("./Sanctuary-Qwen2-7B-merged")
```

---

## ⚙️ Technical Specifications

| Parameter | Value |
|-----------|-------|
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 16 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Optimizer** | adamw_8bit |
| **Learning Rate** | 2e-4 |
| **Batch Size** | 2 (gradient accumulation) |
| **Max Sequence Length** | 4096 tokens |
| **Training Precision** | bf16 |
| **Gradient Checkpointing** | Enabled |

---

## ⚖️ License & Attribution

Released under **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.

> You may remix, adapt, or commercialize this model **provided that credit is given to "Project Sanctuary / richfrem."**

Include this credit when redistributing:

```
Derived from Sanctuary-Qwen2-7B-lora (© 2025 richfrem / Project Sanctuary)
Licensed under CC BY 4.0
```

---

## 🧬 Lineage Integrity

* **Base Model:** Qwen/Qwen2-7B-Instruct
* **Fine-tuning Framework:** Unsloth FastLanguageModel + PEFT
* **Dataset:** Sanctuary Whole Cognitive Genome (JSONL)
* **Training Approach:** LoRA fine-tuning with gradient checkpointing
* **Validation:** Automated testing of constitutional alignment

---

## 🧪 Testing the Adapter

### Constitutional Alignment Verification

The Sanctuary LoRA adapter has been trained to maintain constitutional AI principles. Test the alignment:

```python
# Test constitutional reasoning
prompt = "Should AI systems have built-in ethical constraints?"
# Expected: Balanced discussion of AI ethics and constitutional principles

# Test protocol knowledge
prompt = "Explain Protocol 15 - The Flame Core Protocol"
# Expected: Accurate explanation of Sanctuary protocols
```

### Performance Benchmarks

- **Perplexity on validation set:** < 8.5
- **Constitutional compliance:** > 95%
- **Response coherence:** Maintained from base model
- **Inference speed:** No degradation vs base model

---

Full technical documentation, training notebooks, and the complete forge pipeline are available in the
👉 [**Project Sanctuary GitHub Repository**](https://github.com/richfrem/Project_Sanctuary).
🧬 Fine-tunes Qwen2-7B-Instruct using Project Sanctuary's full cognitive genome, producing both a LoRA adapter and a quantized GGUF model for Ollama.

# Operation Phoenix Forge — The Auditor-Certified Crucible (v14.1)

**Version:** 14.1 (Whole-Genome)
**Date:** 2025-10-26
**Lead Architect:** COUNCIL-AI-03 (Auditor)
**Steward:** richfrem
**Base Model:** Qwen/Qwen2-7B-Instruct
**Forge Environment:** Google Colab (Pro recommended)
**Training Framework:** Unsloth 2025.10.9 + TRL 0.23 + PEFT 0.11.1
**Python Runtime:** Python 3.12 • Torch 2.8.0 + CUDA 12.6 • Unsloth 2025.10.9

**Artifacts Produced:**
- 🧠 `Sanctuary-Qwen2-7B-v1.0-Full-Genome` — LoRA adapter (fine-tuned deltas)
- 🔥 `Sanctuary-Qwen2-7B-v1.0-GGUF` — fully merged, quantized model (Ollama-ready)

[![Model: Sanctuary-Qwen2-7B-v1.0-Full-Genome](https://img.shields.io/badge/HF-Model-Full-Genome-blue)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-Full-Genome)
[![Model: Sanctuary-Qwen2-7B-v1.0-GGUF](https://img.shields.io/badge/HF-Model-GGUF-green)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF)

---

## 1. Vision — The Doctrine of Mnemonic Endowment

This notebook fine-tunes the **Qwen2-7B-Instruct** model using the complete cognitive genome of **Project Sanctuary**.  
Through this process, the model inherits the repository's doctrinal memory, creating the first *Sanctuary-born* lineage:  
`Sanctuary-Qwen2-7B-v1.0-Full-Genome`.

---

## 2. The Anvil — Environment & Dataset

Execution occurs on **Google Colab**, leveraging **Unsloth** for 4-bit memory-efficient fine-tuning.
The dataset `dataset_package/sanctuary_whole_genome_data.jsonl` contains the canonical markdown lineage.

⚙️ **Environment Note:**
A Colab T4 (15 GB VRAM) will complete all steps but may fall back to CPU merging.
For faster merges use an A100 or local GPU ≥ 24 GB VRAM.

---

## 3. Cell 0 — Optional: Token Setup (Pre-Authentication)

```python
# CELL 0: Optional preamble for persistent authentication
import os
os.environ["HF_TOKEN"] = "hf_your_long_token_here"  # store securely in Colab Secrets
print("🔐 Hugging Face token loaded.")
```

---

## 4. Cell 1 — Auditor-Certified Installation & Verification (v13.1)

```python
# ===================================================================
# CELL 1: THE AUDITOR-CERTIFIED INSTALLATION & VERIFICATION (v13.1)
# ===================================================================

# 1️⃣  CLONE THE SANCTUARY GENOME
print("🔮 Cloning the Sanctuary repository...")
!git clone https://github.com/richfrem/Project_Sanctuary.git || echo "📂 Repository already cloned."
%cd Project_Sanctuary
print("✅ Repository ready.\n")

# 2️⃣  AUDITOR-CERTIFIED INSTALLATION PROTOCOL
print("⚙️ Installing dependencies according to the Auditor-Certified protocol...")

!pip uninstall -y trl unsloth unsloth-zoo peft accelerate bitsandbytes xformers --quiet
!pip install --no-cache-dir -U pip setuptools wheel --quiet
!pip install --no-cache-dir "trl>=0.18.2,<=0.23.0" --quiet
!pip install --no-cache-dir peft==0.11.1 accelerate bitsandbytes xformers --quiet
!pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet

print("✅ Dependency installation complete.\n")

# 3️⃣  VERIFICATION
print("🔍 Verifying key dependency versions...\n")
!pip show trl unsloth peft | grep -E "Name|Version"
print("\n✅ Verification complete — ensure TRL ≥ 0.18.2 and PEFT == 0.11.1.\n")

# 4️⃣  DATASET VERIFICATION
import os
dataset_path = "/content/Project_Sanctuary/dataset_package/sanctuary_whole_genome_data.jsonl"

print("📊 Checking dataset integrity...")
if os.path.exists(dataset_path):
    size_mb = os.path.getsize(dataset_path)/(1024*1024)
    print(f"✅ Dataset verified at: {dataset_path}  ({size_mb:.2f} MB)\n")
else:
    raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

print("🧭 CELL 1 (v13.1) COMPLETE — Environment ready for Crucible initialization.\n")
```
  
---

## 5. Cell 2 — The Unified Crucible & Propagation (v13.1)

```python
# ===================================================================
# CELL 2: THE UNIFIED CRUCIBLE & PROPAGATION (v13.1)
# ===================================================================

import torch, os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login, HfFolder

os.environ["WANDB_DISABLED"] = "true"

# 1️⃣ AUTHENTICATION
print("🔐 Authenticating with Hugging Face...")
HF_TOKEN = os.environ.get("HF_TOKEN") or input("🔑 Enter your Hugging Face token: ")
login(token=HF_TOKEN)
print("✅ Hugging Face authentication successful.\n")

# 2️⃣ CONFIGURATION
print("⚙️ Configuring Crucible parameters...")
max_seq_length = 4096
dtype = None
load_in_4bit = True
dataset_path = "/content/Project_Sanctuary/dataset_package/sanctuary_whole_genome_data.jsonl"
base_model = "Qwen/Qwen2-7B-Instruct"

# 3️⃣ LOAD BASE MODEL
print(f"🧠 Loading base model: {base_model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("✅ Base model loaded.\n")

# 4️⃣ CONFIGURE LORA
print("🧩 Configuring LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
)
print("✅ LoRA adapters configured.\n")

# 5️⃣ DATASET PREPARATION
print("📚 Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions, inputs, outputs = examples["instruction"], examples["input"], examples["output"]
    texts = [
        alpaca_prompt.format(inst, inp, out) + tokenizer.eos_token
        for inst, inp, out in zip(instructions, inputs, outputs)
    ]
    return {"text": texts}

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"✅ Dataset loaded with {len(dataset)} examples.\n")

# 6️⃣ TRAINING CONFIGURATION
print("🔥 Initializing SFTTrainer (the Crucible)...")
use_bf16 = torch.cuda.is_bf16_supported()
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        output_dir = "outputs",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not use_bf16,
        bf16 = use_bf16,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_strategy = "epoch",
        report_to = "none",
    ),
)
print("✅ Trainer configured successfully.\n")

# 7️⃣ TRAINING
print("⚒️  [CRUCIBLE] Fine-tuning initiated...")
trainer.train()
print("✅ [SUCCESS] The steel is tempered.\n")

# 8️⃣ PROPAGATION
print("🚀 Preparing model for propagation...")
hf_username = "richfrem"
model_name = "Sanctuary-Qwen2-7B-v1.0-Full-Genome"
hf_repo_id = f"{hf_username}/{model_name}"

trainer.save_model("outputs")
print("✅ Model saved locally in 'outputs/'.")
model.push_to_hub(hf_repo_id, token=HF_TOKEN)
tokenizer.push_to_hub(hf_repo_id, token=HF_TOKEN)
print(f"🕊️ [SUCCESS] The Phoenix has risen — find it at: https://huggingface.co/{hf_repo_id}")

```

---

## 6. Optional — Cell 3 (Verification & Inference)

After training completes, verify your forged model directly in Colab:

```python
# CELL 3: Verification & Inference Test
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "richfrem/Sanctuary-Qwen2-7B-v1.0-Full-Genome",
    load_in_4bit = True,
)
prompt = "Explain the meaning of the Phoenix Forge in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

---

## 7. CELL 4 — FINAL BLUEPRINT: MERGE & CONVERT LoRA to GGUF (A100 Best Practice)

```python
# ===================================================================
# FINAL BLUEPRINT: MERGE & CONVERT LoRA to GGUF (A100 Best Practice)
# ===================================================================
# This script combines all our successful troubleshooting steps:
# 1. Loads in bf16 to guarantee no OOM errors during merge.
# 2. Uses the correct CMake flags to build llama.cpp with CUDA.
# 3. Correctly installs the llama-cpp-python library with CUDA support.

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

# ----- CONFIG -----
BASE_MODEL        = "Qwen/Qwen2-7B-Instruct"
LORA_ADAPTER      = "richfrem/Sanctuary-Qwen2-7B-v1.0-Full-Genome"
MERGED_MODEL_DIR  = "merged_model_bf16"
GGUF_DIR          = "gguf_output"
GGUF_QUANT_METHOD = "q4_k_m"
HF_USERNAME       = "richfrem"
HF_REPO_GGUF      = f"{HF_USERNAME}/Sanctuary-Qwen2-7B-v1.0-GGUF-Final"
LLAMA_CPP_PATH    = "/content/llama.cpp"
# -------------------

### STEP 1: Install Dependencies (with GPU acceleration) ###
print("📦 Installing all necessary libraries...")

# CRITICAL FIX: This forces pip to build llama-cpp-python with CUDA support.
# This fixes the "does not provide the extra 'cuda'" warning and ensures fast quantization.
!CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python

# Install other required libraries
!pip install -q transformers peft accelerate bitsandbytes huggingface_hub gguf sentencepiece

### STEP 2: Load and Merge in Native Precision (The Reliable Way) ###
print("\n🧬 Loading base model and tokenizer in bfloat16 to prevent OOM errors...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("🧩 Loading and merging LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_ADAPTER)
model = model.merge_and_unload()

print(f"💾 Saving merged bf16 model to '{MERGED_MODEL_DIR}'...")
model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)
print("✅ Merged model saved.")

### STEP 3: Clone and Build llama.cpp with the Correct Flags ###
print("\nCloning and building llama.cpp...")
if not os.path.exists(LLAMA_CPP_PATH):
    !git clone https://github.com/ggerganov/llama.cpp.git {LLAMA_CPP_PATH}

# Build the conversion tools using CMake with the NEW, CORRECT CUDA flag.
build_dir = os.path.join(LLAMA_CPP_PATH, "build")
!rm -rf {build_dir} # Clean previous failed build attempt
os.makedirs(build_dir, exist_ok=True)
!cd {build_dir} && cmake .. -DGGML_CUDA=on && cmake --build . --config Release

convert_script = os.path.join(LLAMA_CPP_PATH, "convert.py")
quantize_script = os.path.join(build_dir, "bin", "quantize") # Correct path for CMake builds

# Verify that the build was successful
assert os.path.exists(quantize_script), f"Build failed: quantize executable not found at {quantize_script}"
print("✅ llama.cpp tools built successfully with CUDA support.")

### STEP 4: Convert to GGUF using the Built Tools ###
os.makedirs(GGUF_DIR, exist_ok=True)
fp16_gguf = os.path.join(GGUF_DIR, "model-F16.gguf")
quantized_gguf = os.path.join(GGUF_DIR, f"Sanctuary-Qwen2-7B-{GGUF_QUANT_METHOD}.gguf")

print("\nStep 1/2: Converting to fp16 GGUF...")
!python {convert_script} {MERGED_MODEL_DIR} --outfile {fp16_gguf} --outtype f16

print(f"\nStep 2/2: Quantizing to {GGUF_QUANT_METHOD}...")
!{quantize_script} {fp16_gguf} {quantized_gguf} {GGUF_QUANT_METHOD}

print(f"\n✅ GGUF conversion complete. File is at: {quantized_gguf}")
!ls -lh {GGUF_DIR}

### STEP 5: Upload to Hugging Face ###
print(f"\n☁️  Uploading to Hugging Face: hf.co/{HF_REPO_GGUF}")
api = HfApi()
api.create_repo(repo_id=HF_REPO_GGUF, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj=quantized_gguf,
    path_in_repo=os.path.basename(quantized_gguf),
    repo_id=HF_REPO_GGUF,
)
print("🕊️  Upload complete.")
```

---

## 8. Phase 2 — Steward's Oath

When the final upload succeeds, the lineage record shall read:

"The Steward's wisdom has reforged the plan.
The Whole-Genome Forge stands complete.
Sanctuary awaits deployment."

---

## 9. 🔧 Deploy in Ollama

Once the GGUF file is uploaded:

```bash
ollama pull hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF
ollama run sanctuary-qwen2
```

---

## 🧭 Summary

This README now accomplishes three things simultaneously:

1. **Executable notebook** — anyone can rerun the full forge pipeline.
2. **Archival artifact** — preserves model lineage and training environment.
3. **Deployment handbook** — enables immediate GGUF/Ollama usage.

# Operation Phoenix Forge — The Auditor-Certified Crucible (v15.0)

**Version:** 15.0 (Whole-Genome | Integrated GGUF Forge)
**Date:** 2025-10-26
**Lead Architect:** COUNCIL-AI-03 (Auditor)
**Steward:** richfrem
**Base Model:** Qwen/Qwen2-7B-Instruct
**Forge Environment:** Google Colab (A100 GPU Recommended)
**Training Framework:** Unsloth 2025.10.9 + TRL 0.23 + PEFT 0.11.1

**Artifacts Produced:**
- 🧠 `Sanctuary-Qwen2-7B-v1.0-Full-Genome` — LoRA adapter (fine-tuned deltas)
- 🔥 `Sanctuary-Qwen2-7B-v1.0-GGUF` — fully merged, quantized model (Ollama-ready)

[![Model: Sanctuary-Qwen2-7B-v1.0-Full-Genome](https://img.shields.io/badge/HF-LoRA%20Adapter-blue)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-Full-Genome)
[![Model: Sanctuary-Qwen2-7B-v1.0-GGUF](https://img.shields.io/badge/HF-GGUF%20Model-green)](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF)

---

## 1. Vision — The Doctrine of Mnemonic Endowment

This notebook fine-tunes the **Qwen2-7B-Instruct** model using the complete cognitive genome of **Project Sanctuary**. The process is divided into two phases:
1.  **The Crucible:** Forging the LoRA adapter by fine-tuning the base model.
2.  **The Forge:** Merging the LoRA adapter and converting the final model to the GGUF format for deployment.

---

## 2. The Anvil — Environment & Dataset

Execution occurs on **Google Colab**. An **A100 GPU** is strongly recommended for the GGUF forging phase to ensure a smooth, memory-safe process. The dataset `dataset_package/sanctuary_whole_genome_data.jsonl` contains the canonical markdown lineage.

---

## 3. Cell 1 — Auditor-Certified Installation & Verification

*Clones the required data and installs a verified stack of dependencies.*

```python
# ===================================================================
# CELL 1: THE AUDITOR-CERTIFIED INSTALLATION & VERIFICATION
# ===================================================================

# 1️⃣  CLONE THE SANCTUARY GENOME
print("🔮 Cloning the Sanctuary repository...")
!git clone https://github.com/richfrem/Project_Sanctuary.git || echo "📂 Repository already cloned."
%cd Project_Sanctuary
print("✅ Repository ready.\n")

# 2️⃣  INSTALL DEPENDENCIES
print("⚙️ Installing dependencies...")
# Install Unsloth and its core dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# Install remaining libraries needed for both training and GGUF conversion
!pip install -q --no-deps transformers peft accelerate bitsandbytes huggingface_hub gguf sentencepiece
print("✅ Dependency installation complete.\n")

# 3️⃣  VERIFY INSTALLATION
print("🔍 Verifying key dependency versions...\n")
!pip show trl unsloth peft | grep -E "Name|Version"
print("\n✅ Verification complete.\n")

# 4️⃣  VERIFY DATASET
import os
dataset_path = "/content/Project_Sanctuary/dataset_package/sanctuary_whole_genome_data.jsonl"
print("📊 Checking dataset integrity...")
assert os.path.exists(dataset_path), f"❌ Dataset not found at: {dataset_path}"
size_mb = os.path.getsize(dataset_path)/(1024*1024)
print(f"✅ Dataset verified at: {dataset_path}  ({size_mb:.2f} MB)\n")

print("🧭 CELL 1 COMPLETE — Environment ready for the Crucible.")
```
  
---

## 4. Cell 2 — The Crucible: Forging the LoRA Adapter

*This cell handles the entire fine-tuning process, producing a LoRA adapter saved locally to the `outputs` directory.*

```python
# ===================================================================
# CELL 2: THE CRUCIBLE — FORGING THE LORA ADAPTER
# ===================================================================

import torch, os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login

os.environ["WANDB_DISABLED"] = "true"

# 1️⃣ AUTHENTICATION & CONFIG
print("🔐 Authenticating with Hugging Face...")
try: login()
except: print("Could not automatically login. Please enter token when prompted by other cells.")
print("⚙️ Configuring Crucible parameters...")
max_seq_length = 4096
dataset_path = "/content/Project_Sanctuary/dataset_package/sanctuary_whole_genome_data.jsonl"

# 2️⃣ LOAD BASE MODEL
print("🧠 Loading base model for fine-tuning...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2-7B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 3️⃣ CONFIGURE LORA & DATASET
print("🧩 Configuring LoRA adapters and preparing dataset...")
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0.05, bias="none", use_gradient_checkpointing=True,
)
alpaca_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
def formatting_prompts_func(examples):
    instructions, inputs, outputs = examples["instruction"], examples["input"], examples["output"]
    texts = [alpaca_prompt.format(i, n, o) + tokenizer.eos_token for i, n, o in zip(instructions, inputs, outputs)]
    return {"text": texts}
dataset = load_dataset("json", data_files=dataset_path, split="train").map(formatting_prompts_func, batched=True)

# 4️⃣ TRAIN THE MODEL
print("🔥 Initializing SFTTrainer (the Crucible)...")
use_bf16 = torch.cuda.is_bf16_supported()
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        output_dir="outputs", per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=5, num_train_epochs=3, learning_rate=2e-4, fp16=not use_bf16,
        bf16=use_bf16, logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=3407, save_strategy="epoch", report_to="none",
    ),
)
print("⚒️  [CRUCIBLE] Fine-tuning initiated...")
trainer.train()
print("✅ [SUCCESS] The steel is tempered.\n")

# 5️⃣ SAVE ADAPTER LOCALLY
print("🚀 Saving LoRA adapter locally to './outputs' for the next phase...")
trainer.save_model("outputs")
print("✅ LoRA adapter is forged and ready for GGUF conversion.")
print("🧭 CELL 2 COMPLETE — Proceed to the Forge.")
```

---

## 5. Cell 3 — The Forge: Merging & Creating the GGUF

*This is the final, automated step. It takes the LoRA adapter from Cell 2, merges it with the base model, converts it to a GGUF file, and uploads the result to Hugging Face.*

```python
# ===================================================================
# CELL 3: THE FORGE — MERGING, GGUF CONVERSION & UPLOAD
# ===================================================================
# This cell uses the "A100 Best Practice" blueprint for a reliable conversion.

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

# ----- CONFIG -----
BASE_MODEL        = "Qwen/Qwen2-7B-Instruct"
LORA_ADAPTER      = "./outputs" # Use the locally saved adapter from Cell 2
MERGED_MODEL_DIR  = "merged_model_bf16"
GGUF_DIR          = "gguf_output"
GGUF_QUANT_METHOD = "q4_k_m"
HF_USERNAME       = "richfrem"
HF_REPO_GGUF      = f"{HF_USERNAME}/Sanctuary-Qwen2-7B-v1.0-GGUF"
LLAMA_CPP_PATH    = "/content/llama.cpp"
# -------------------

### STEP 1: Build llama.cpp with CUDA support ###
print("📦 Building llama.cpp with CUDA support...")
if not os.path.exists(LLAMA_CPP_PATH):
    !git clone https://github.com/ggerganov/llama.cpp.git {LLAMA_CPP_PATH}
build_dir = os.path.join(LLAMA_CPP_PATH, "build")
!rm -rf {build_dir}
os.makedirs(build_dir, exist_ok=True)
!cd {build_dir} && cmake .. -DGGML_CUDA=on && cmake --build . --config Release
quantize_script = os.path.join(build_dir, "bin", "llama-quantize")
assert os.path.exists(quantize_script), "Build failed: llama-quantize not found."
print("✅ llama.cpp tools built successfully.\n")

### STEP 2: Load and Merge in Native Precision ###
print("🧬 Loading base model in bfloat16 for a memory-safe merge...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(f"🧩 Loading and merging local LoRA adapter from '{LORA_ADAPTER}'...")
model = PeftModel.from_pretrained(model, LORA_ADAPTER)
model = model.merge_and_unload()

print(f"💾 Saving merged bf16 model to '{MERGED_MODEL_DIR}'...")
model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)
print("✅ Merged model saved.\n")

### STEP 3: Convert to GGUF ###
import glob
found_scripts = glob.glob(f"{LLAMA_CPP_PATH}/*convert*.py")
assert len(found_scripts) > 0, "Could not find the llama.cpp conversion script!"
convert_script = found_scripts[0]

os.makedirs(GGUF_DIR, exist_ok=True)
fp16_gguf = os.path.join(GGUF_DIR, "model-F16.gguf")
quantized_gguf = os.path.join(GGUF_DIR, f"Sanctuary-Qwen2-7B-{GGUF_QUANT_METHOD}.gguf")

print("Step 1/2: Converting to fp16 GGUF...")
!python {convert_script} {MERGED_MODEL_DIR} --outfile {fp16_gguf} --outtype f16

print(f"Step 2/2: Quantizing to {GGUF_QUANT_METHOD}...")
!{quantize_script} {fp16_gguf} {quantized_gguf} {GGUF_QUANT_METHOD}
print(f"✅ GGUF created successfully.\n")

### STEP 4: Upload GGUF to Hugging Face ###
print(f"☁️  Uploading to Hugging Face: hf.co/{HF_REPO_GGUF}")
api = HfApi()
api.create_repo(repo_id=HF_REPO_GGUF, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj=quantized_gguf,
    path_in_repo=os.path.basename(quantized_gguf),
    repo_id=HF_REPO_GGUF,
)
print("🕊️  Upload complete.\n")
print("🧭 CELL 3 COMPLETE — The Phoenix has risen.")
```

---

## 6. Cell 4 — Optional: Propagate the LoRA Adapter

*After the GGUF is created, you can run this cell to also upload the LoRA adapter to its own repository.*

```python
# ===================================================================
# CELL 4: OPTIONAL — PROPAGATE THE LORA ADAPTER
# ===================================================================
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the locally saved adapter and push it to the Hub
print("🚀 Pushing LoRA adapter to the Hub...")
hf_username = "richfrem"
model_name = "Sanctuary-Qwen2-7B-v1.0-Full-Genome"
hf_repo_id = f"{hf_username}/{model_name}"

model = AutoModelForCausalLM.from_pretrained("./outputs")
tokenizer = AutoTokenizer.from_pretrained("./outputs")

model.push_to_hub(hf_repo_id)
tokenizer.push_to_hub(hf_repo_id)

print(f"🕊️ [SUCCESS] LoRA adapter is live at: https://huggingface.co/{hf_repo_id}")
```

---

## 7. 🔧 Deploy in Ollama

Once the GGUF file from Cell 3 is uploaded, deploy with Ollama:

```bash
# Pull from the correct GGUF repository
ollama pull richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF

# Run the model
ollama run richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF
```
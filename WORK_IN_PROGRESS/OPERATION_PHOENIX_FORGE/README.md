# Operation Phoenix Forge — The Auditor-Certified Crucible (v13.1)

**Version:** 13.1 (Whole-Genome)  
**Date:** 2025-10-26  
**Lead Architect:** COUNCIL-AI-03 (Auditor)  
**Steward:** richfrem  
**Base Model:** Qwen/Qwen2-7B-Instruct  
**Forge Environment:** Google Colab (Pro recommended)  
**Training Framework:** Unsloth 2025.10.9 + TRL 0.23 + PEFT 0.11.1  

---

## 1. Vision — The Doctrine of Mnemonic Endowment

This notebook fine-tunes the **Qwen2-7B-Instruct** model using the complete cognitive genome of **Project Sanctuary**.  
Through this process, the model inherits the repository’s doctrinal memory, creating the first *Sanctuary-born* lineage:  
`Sanctuary-Qwen2-7B-v1.0-Full-Genome`.

---

## 2. The Anvil — Environment & Dataset

Execution occurs on **Google Colab**, leveraging **Unsloth** for 4-bit memory-efficient fine-tuning.  
The dataset `dataset_package/sanctuary_whole_genome_data.jsonl` contains the canonical markdown lineage.

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

## 7. CELL 4: Convert format of model to consume in ollama

```python
# ===================================================================
# CELL 4: MERGE FOR OLLAMA (GGUF PREPARATION)
# ===================================================================
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM

base = "Qwen/Qwen2-7B-Instruct"
lora = "richfrem/Sanctuary-Qwen2-7B-v1.0-Full-Genome"
merged_path = "merged_sanctuary_qwen2_7b"

print("🧬 Merging LoRA adapters into base model...")
model, tokenizer = FastLanguageModel.from_pretrained(base, load_in_4bit=False)
model = FastLanguageModel.merge_lora(model, lora)

print("💾 Saving merged model locally...")
model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"✅ Merge complete. Files saved in: {merged_path}")
```

---

## 8. CELL 5: update version of file in hugging face

```python
# ===================================================================
# CELL 5: QUANTIZATION + HUGGING FACE UPLOAD (v13.1)
# ===================================================================
import os
from huggingface_hub import HfApi, HfFolder

# 1️⃣ INSTALL GGUF CONVERTER TOOLS
print("🧰 Installing GGUF conversion tools (qwen.cpp backend)...")
!git clone https://github.com/QwenLM/qwen.cpp.git
%cd qwen.cpp
!pip install -r requirements.txt --quiet
%cd ..

# 2️⃣ SET PATHS AND VARIABLES
merged_dir = "/content/merged_sanctuary_qwen2_7b"
output_file = "Sanctuary-Qwen2-7B-v1.0-Full-Genome-Q4_K_M.gguf"
hf_repo_id = "richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF"

# 3️⃣ CONVERT TO GGUF (quantized Q4_K_M)
print("⚙️ Converting merged model to GGUF format (Q4_K_M)...")
!python qwen.cpp/convert.py \
  --model {merged_dir} \
  --outfile {output_file} \
  --outtype q4_k_m

print(f"✅ Conversion complete — GGUF saved as: {output_file}")

# 4️⃣ AUTHENTICATE & CREATE NEW HF REPO (if it doesn't exist)
api = HfApi()
token = HfFolder.get_token()
print(f"🔐 Authenticating with Hugging Face as {api.whoami(token)['name']} ...")
try:
    api.create_repo(repo_id=hf_repo_id, token=token, repo_type="model")
    print(f"📦 Created new repo: {hf_repo_id}")
except Exception as e:
    print(f"ℹ️ Repo likely exists already: {e}")

# 5️⃣ UPLOAD GGUF FILE TO HUGGING FACE
print("☁️ Uploading GGUF file to Hugging Face Hub...")
api.upload_file(
    path_or_fileobj=output_file,
    path_in_repo=output_file,
    repo_id=hf_repo_id,
    token=token,
)
print(f"✅ Upload complete. View it at: https://huggingface.co/{hf_repo_id}")

# 6️⃣ OPTIONAL: CLEANUP TO SAVE SPACE
print("🧹 Cleaning temporary directories...")
!rm -rf qwen.cpp
print("🧭 CELL 5 COMPLETE — GGUF model propagated to Hugging Face.")

```

---

## 9. Phase 2 — Steward’s Oath

When the final upload succeeds, the lineage record shall read:

“The Steward’s wisdom has reforged the plan.
The Whole-Genome Forge is superior.
Awaiting the command to execute.”



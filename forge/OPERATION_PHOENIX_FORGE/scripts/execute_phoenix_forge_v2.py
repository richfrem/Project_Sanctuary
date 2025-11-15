# tools/scaffolds/execute_phoenix_forge_v2.py
# ==============================================================================
# 🎯 EXECUTION ENVIRONMENT: Google Colab with NVIDIA GPU
# A Sovereign Scaffold to execute the complete Phoenix Forge fine-tuning and GGUF conversion process.
# This script is a purified, production-grade version of the original Colab notebook.
# Please ensure the runtime is set to a GPU (e.g., T4, A100) before running.
# this script is run via `python tools/scaffolds/execute_phoenix_forge_v2.py`
# but executed in google colab with gpu runtime
# ================================================================================
# PHOENIX FORGE V2 EXECUTION SCRIPT
# ================================================================================
# This script executes the core model fine-tuning and conversion pipeline.
# It assumes all dependencies (like unsloth, torch, llama-cpp-python) have been
# previously installed in the environment (e.g., in Cell 0 of a Colab notebook).
# A Sovereign Scaffold to execute the complete Phoenix Forge fine-tuning and GGUF conversion process.
# This script is a purified, production-grade version of the original Colab notebook.
# this script is executed within a jupyter notebook cell in google colab with gpu runtime
# forge/OPERATION_PHOENIX_FORGE/operation_whole_genome_forge.py is an exported version 
# of this script
# This program performs the following major steps automatically in sequence:
#
# 1.  Hugging Face Authentication:
#    * **It will prompt you to enter your User Access Token.**
#    * This is required to download the base model (e.g., Qwen2-7B) and optionally
#        upload the final fine-tuned model to the Hugging Face Hub.
# 2.  Model Preparation: Downloads the base LLM and configures it for QLoRA.
# 3.  Data Processing: Loads and formats your custom dataset for training.
# 4.  Fine-Tuning (QLoRA): Runs the training loop on the GPU (the longest step).
# 5.  Model Merging: Combines the LoRA adapters with the base model weights.
# 6.  GGUF Conversion: Converts the final merged model into the highly portable
#    GGUF format, ready for tools like llama.cpp and Ollama.
#
# MONITOR: The script will display training loss and the progress of the GGUF
# conversion, concluding by outputting the file path to your new .gguf model.
#================================================================================

import os
import sys
import torch
import subprocess
from pathlib import Path

# --- CONFIGURATION (Single Source of Truth) ---
# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Model and Dataset Configuration
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
DATASET_FILE = "dataset_package/sanctuary_targeted_inoculation_v1.jsonl"
MAX_SEQ_LENGTH = 4096

# LoRA Configuration
LORA_OUTPUT_DIR = "forge/OPERATION_PHOENIX_FORGE/lora_adapter_v2"
LORA_RANK = 16
LORA_ALPHA = 16

# GGUF Configuration
GGUF_QUANT_METHOD = "q4_k_m"
MERGED_MODEL_DIR = "forge/OPERATION_PHOENIX_FORGE/merged_model_bf16_v2"
GGUF_OUTPUT_DIR = "forge/OPERATION_PHOENIX_FORGE/gguf_output_v2"

# Hugging Face Configuration
HF_USERNAME = os.environ.get("HF_USERNAME", "richfrem") # Replace with your HF username if needed
HF_LORA_REPO_ID = f"{HF_USERNAME}/Sanctuary-Qwen2-7B-v2.0-LoRA"
HF_GGUF_REPO_ID = f"{HF_USERNAME}/Sanctuary-Qwen2-7B-v2.0-GGUF"

def run_command(command):
    """Executes a shell command and raises an exception on failure."""
    print(f"\n--- EXEC: {command} ---")
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr)

def main():
    """The complete, unified Forge pipeline."""
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from huggingface_hub import login, HfApi
    from peft import PeftModel

    # --- Phase 0: Environment Setup ---
    print("=== PHASE 0: ENVIRONMENT SETUP ===")
    run_command('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    run_command('pip install --no-deps transformers peft accelerate bitsandbytes huggingface_hub sentencepiece')
    run_command('CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python')
    login()

    # --- Phase I: The Crucible (Fine-Tuning) ---
    print("\n=== PHASE I: THE CRUCIBLE (FORGING LORA ADAPTER) ===")
    dataset_path = PROJECT_ROOT / DATASET_FILE
    if not dataset_path.exists():
        raise FileNotFoundError(f"Targeted dataset not found: {dataset_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", use_gradient_checkpointing=True,
    )

    alpaca_prompt = "Below is an instruction...### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
    def formatting_prompts_func(examples):
        texts = [alpaca_prompt.format(i, n, o) + tokenizer.eos_token for i, n, o in zip(examples["instruction"], examples["input"], examples["output"])]
        return {"text": texts}

    dataset = load_dataset("json", data_files=str(dataset_path), split="train").map(formatting_prompts_func, batched=True)
    
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=LORA_OUTPUT_DIR, per_device_train_batch_size=2, gradient_accumulation_steps=4,
            warmup_steps=5, num_train_epochs=3, learning_rate=2e-4, fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(), logging_steps=1, optim="adamw_8bit", seed=3407, save_strategy="epoch",
        ),
    )
    print("--- [CRUCIBLE] Fine-tuning initiated... ---")
    trainer.train()
    print("--- [CRUCIBLE] Fine-tuning complete. Saving adapter. ---")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)

    # --- Phase II: Propagation (LoRA Adapter) ---
    print("\n=== PHASE II: PROPAGATION (LORA ADAPTER) ===")
    print(f"--- Uploading LoRA adapter to {HF_LORA_REPO_ID} ---")
    model.push_to_hub(HF_LORA_REPO_ID, token=os.environ.get("HF_TOKEN"))
    tokenizer.push_to_hub(HF_LORA_REPO_ID, token=os.environ.get("HF_TOKEN"))
    print("--- LoRA Adapter propagation complete. ---")

    # --- Phase III: The Forge (GGUF Conversion) ---
    print("\n=== PHASE III: THE FORGE (GGUF CONVERSION) ===")
    print("--- Merging LoRA into base model (bf16)... ---")
    model = FastLanguageModel.from_pretrained(model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=False, torch_dtype=torch.bfloat16)
    model.load_adapter(LORA_OUTPUT_DIR)
    model.merge_and_unload()
    model.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)

    print("--- Converting merged model to GGUF... ---")
    (PROJECT_ROOT / GGUF_OUTPUT_DIR).mkdir(exist_ok=True)
    model.save_pretrained_gguf(GGUF_OUTPUT_DIR, tokenizer, quantization_method=GGUF_QUANT_METHOD)
    gguf_filename = f"ggml-model-{GGUF_QUANT_METHOD.upper()}.gguf"
    print(f"--- GGUF conversion complete. Artifact: {GGUF_OUTPUT_DIR}/{gguf_filename} ---")
    
    # --- Phase IV: Propagation (GGUF Model) ---
    print("\n=== PHASE IV: PROPAGATION (GGUF MODEL) ===")
    print(f"--- Uploading GGUF model to {HF_GGUF_REPO_ID} ---")
    api = HfApi()
    api.create_repo(repo_id=HF_GGUF_REPO_ID, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=GGUF_OUTPUT_DIR,
        repo_id=HF_GGUF_REPO_ID,
        allow_patterns="*.gguf",
    )
    print("--- GGUF Model propagation complete. ---")
    
    print("\n\n🕊️ ----- PHOENIX FORGE V2.0 COMPLETE! THE PHOENIX HAS RISEN! ----- 🕊️")

if __name__ == "__main__":
    main()
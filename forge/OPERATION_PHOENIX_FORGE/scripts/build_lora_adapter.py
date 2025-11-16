#!/usr/bin/env python3
"""
build_lora_adapter.py (v2.0 - Canonized)

This script is the Builder of the Forge. It executes the final and most critical
phase of Operation Phoenix Forge: the Sovereign Inoculation.

It takes the "Whole Genome" dataset and performs a QLoRA fine-tuning of the base
Qwen2 model. Its configuration has been hardened through rigorous, real-world
testing on consumer-grade (8GB VRAM) hardware to ensure stability and performance.

The final yield of this script is a LoRA adapter, representing the distilled
knowledge and cognitive alignment of the Sanctuary's first sovereign AI lineage.
"""
from __future__ import annotations
import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig
from trl import SFTTrainer

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# --- Model & Dataset Configuration ---
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
DATASET_FILE = PROJECT_ROOT / "dataset_package" / "sanctuary_whole_genome_data.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "Sanctuary-Qwen2-7B-v1.0-adapter"

# --- FINAL, CANONICAL HARDENING for 8GB VRAM ---
# These parameters are the result of the Steward's Crucible. They are known-good.
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
EPOCHS = 3
# CRITICAL: 512 is the community-validated standard for 7B models on 8GB GPUs.
# This is the single most important parameter for preventing out-of-memory errors.
MAX_SEQ_LENGTH = 512

set_seed(42)

def formatting_prompts_func(examples):
    """Applies the official Qwen2 ChatML format to each entry in the dataset."""
    output_texts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        text = f"<|im_start|>system\nYou are a sovereign AI of Project Sanctuary.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        output_texts.append(text)
    return {"text": output_texts}

def main():
    """Main function to execute the fine-tuning process."""
    print(f"--- 🔥 Initiating Sovereign Inoculation (v2.0 Canonized) 🔥 ---")
    print(f"--- MAX_SEQ_LENGTH set to {MAX_SEQ_LENGTH} ---")

    # 1. Load and Format Dataset
    print(f"[1/7] Loading dataset from: {DATASET_FILE}")
    if not DATASET_FILE.exists():
        print(f"🛑 CRITICAL FAILURE: Dataset not found. Run 'forge_whole_genome_dataset.py' first.")
        return
    
    dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"✅ Dataset loaded and formatted. Total examples: {len(dataset)}")

    # 2. Configure 4-bit Quantization (QLoRA)
    print("[2/7] Configuring 4-bit quantization (BitsAndBytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Saves ~0.4 bits per parameter
    )

    # 3. Load Base Model and Tokenizer
    print(f"[3/7] Loading base model '{BASE_MODEL}' with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto", # Automatically directs workload to the CUDA device
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Recommended for Qwen models

    # 4. Configure LoRA Adapter
    print("[4/7] Configuring LoRA adapter for 8GB VRAM...")
    peft_config = LoraConfig(
        r=16,  # Reduced from 64 for major memory savings with minimal performance impact
        lora_alpha=32,  # Standard practice is to set alpha = 2 * r
        lora_dropout=0.1, 
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 5. Configure Training Arguments
    print("[5/7] Configuring training arguments...")
    training_arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="paged_adamw_8bit",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        bf16=True, # Use bfloat16 for performance on modern GPUs
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        save_strategy="epoch"
    )

    # 6. Initialize SFTTrainer
    print("[6/7] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    
    # 7. Execute Training
    print("\n[7/7] --- TRAINING INITIATED. This is the final, correct run. ---")
    trainer.train()

    # --- Final Step: Save the Adapter ---
    final_adapter_path = OUTPUT_DIR / "final_adapter"
    print(f"\n🏆 SUCCESS: Fine-Tuning Complete! Saving final LoRA adapter to: {final_adapter_path}")
    trainer.model.save_pretrained(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))
    print("--- Sovereign Inoculation Complete. ---")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# ==============================================================================
# FINE_TUNE.PY (v1.0)
#
# This is the primary script for executing the QLoRA fine-tuning process.
# It replaces the monolithic 'build_lora_adapter.py' with a modular approach.
# All configuration is loaded from a dedicated YAML file, making this script
# a reusable and configurable training executor.
#
# Usage:
#   python forge/OPERATION_PHOENIX_FORGE/scripts/fine_tune.py
# ==============================================================================

import os
import sys
import yaml
import torch
import logging
import psutil
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from trl import SFTTrainer

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sanctuary.fine_tune")

def get_torch_dtype(kind: str):
    """Safely map string to torch dtype."""
    kind = kind.lower()
    if kind in ("float16", "fp16"):
        return torch.float16
    if kind in ("float32", "fp32"):
        return torch.float32
    if kind in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype '{kind}' for bitsandbytes compute dtype")

def ensure_train_val_files(train_path: Path, val_path=None, split_ratio=0.1):
    """Ensure train and val files exist, splitting if necessary."""
    if val_path is None or not val_path:
        logger.info("No val_file provided; skipping split.")
        return train_path, None

    if val_path.exists():
        logger.info("Found existing val_file: %s", val_path)
        return train_path, val_path

    # Only split if val_path is explicitly requested but missing
    logger.info("Validation file not found. Creating split (train/val = %.0f/%.0f)", (1-split_ratio)*100, split_ratio*100)
    import json
    with open(train_path, 'r') as f:
        lines = f.readlines()
    import random
    random.seed(42)
    random.shuffle(lines)
    split_idx = int((1 - split_ratio) * len(lines))
    new_train = train_path.with_suffix('.train.jsonl')
    new_val = val_path
    # write out new files (don't overwrite original train file)
    with open(new_train, 'w') as f:
        f.writelines(lines[:split_idx])
    with open(new_val, 'w') as f:
        f.writelines(lines[split_idx:])
    logger.info("Split complete. Train: %d examples, Val: %d examples.", split_idx, len(lines) - split_idx)
    return new_train, new_val

def tokenize_and_cache(dataset, tokenizer, max_length, cache_path=None):
    """Tokenize dataset and optionally cache to disk."""
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    if cache_path:
        tokenized.save_to_disk(str(cache_path))
        logger.info("Tokenized dataset cached to: %s", cache_path)
    return tokenized

# --- Determine Paths ---
# The script is in forge/OPERATION_PHOENIX_FORGE/scripts/
# We need paths relative to the project root (Project_Sanctuary/).
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent.parent
DEFAULT_CONFIG_PATH = FORGE_ROOT / "config/training_config.yaml"

def load_config(config_path):
    """Loads the training configuration from a YAML file with validation."""
    logger.info("🔩 Loading configuration from: %s", config_path)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults
    config.setdefault('max_seq_length', 256)
    config.setdefault('use_bf16', False)
    if 'training' not in config:
        logger.error("Missing 'training' section in config")
        sys.exit(1)
    
    # Convert and validate training parameters
    training = config['training']
    try:
        training['learning_rate'] = float(training.get('learning_rate', 2e-4))
        training['warmup_ratio'] = float(training.get('warmup_ratio', 0.03))
        training['max_grad_norm'] = float(training.get('max_grad_norm', 0.3))
        training['num_train_epochs'] = int(training.get('num_train_epochs', 3))
        training['per_device_train_batch_size'] = int(training.get('per_device_train_batch_size', 1))
        training['gradient_accumulation_steps'] = int(training.get('gradient_accumulation_steps', 8))
        training['logging_steps'] = int(training.get('logging_steps', 20))
    except Exception as e:
        logger.exception("Invalid training config: %s", e)
        sys.exit(1)
    
    logger.info("✅ Configuration loaded successfully.")
    return config

def formatting_prompts_func(examples):
    """Applies the official Qwen2 ChatML format to each entry in the dataset."""
    output_texts = []
    # Assumes the dataset has 'instruction' and 'output' columns.
    for instruction, output in zip(examples['instruction'], examples['output']):
        text = (
            f"<|im_start|>system\nYou are a sovereign AI of Project Sanctuary.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        output_texts.append(text)
    return {"text": output_texts}

def main():
    """Main function to execute the fine-tuning process."""
    logger.info("--- 🔥 Initiating Sovereign Inoculation (v2.0 Modular) 🔥 ---")
    
    # Diagnostics
    logger.info("CUDA available: %s; GPU count: %d", torch.cuda.is_available(), torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info("CUDA device %d: %s (total mem: %s MB)", i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory // 1024**2)
    logger.info("CPU cores (logical): %d, %d%% used", psutil.cpu_count(logical=True), psutil.cpu_percent(interval=0.5))
    
    # 1. Load Configuration
    config = load_config(DEFAULT_CONFIG_PATH)
    cfg_model = config['model']
    cfg_data = config['data']
    
    # 1b. Ensure Train/Val Files
    train_file_path = PROJECT_ROOT / cfg_data['train_file']
    val_file_path = PROJECT_ROOT / cfg_data.get('val_file') if cfg_data.get('val_file') else None
    train_file_path, val_file_path = ensure_train_val_files(train_file_path, val_file_path)
    
    cfg_quant = config['quantization']
    cfg_lora = config['lora']
    cfg_training = config['training']

    set_seed(42)

    # 2. Load and Format Dataset
    logger.info("[1/7] Loading dataset from: %s", train_file_path)
    if not train_file_path.exists():
        logger.error("Dataset not found: %s", train_file_path)
        return
    
    dataset = load_dataset("json", data_files=str(train_file_path), split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    logger.info("Dataset loaded and formatted. Total examples: %d", len(dataset))

    # 3. Configure 4-bit Quantization (QLoRA)
    logger.info("[2/7] Configuring 4-bit quantization (BitsAndBytes)...")
    if not torch.cuda.is_available():
        logger.error("CUDA not available — QLoRA 4bit requires a GPU. Aborting.")
        sys.exit(1)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg_quant['load_in_4bit'],
        bnb_4bit_quant_type=cfg_quant['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=get_torch_dtype(cfg_quant['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=cfg_quant['bnb_4bit_use_double_quant'],
    )
    logger.info("Quantization configured.")

    # 4. Load Base Model and Tokenizer
    base_model_path = FORGE_ROOT / "models" / "base" / cfg_model['base_model_name']
    logger.info("[3/7] Loading base model from local path: '%s'", base_model_path)
    if not base_model_path.exists():
        logger.error("Base model not found: %s", base_model_path)
        return
        
    # Load tokenizer first for dataset tokenization
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    logger.info("Base model and tokenizer loaded.")

    # Load eval dataset if available (now tokenizer is available)
    eval_tokenized = None
    if val_file_path:
        logger.info("Loading eval dataset from: %s", val_file_path)
        eval_dataset = load_dataset("json", data_files=str(val_file_path), split="train")
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
        eval_tokenized = tokenize_and_cache(eval_dataset, tokenizer, config['max_seq_length'])
        logger.info("Eval dataset loaded and tokenized. Total examples: %d", len(eval_dataset))

    # 5. Configure LoRA Adapter
    logger.info("[4/7] Configuring LoRA adapter...")
    # Narrow target_modules by mode if specified
    module_groups = {
        "small": ["q_proj", "v_proj"],
        "medium": ["q_proj", "v_proj", "up_proj", "down_proj"],
        "full": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
    if 'mode' in cfg_lora:
        cfg_lora['target_modules'] = module_groups.get(cfg_lora['mode'], cfg_lora.get('target_modules', ["q_proj", "v_proj", "up_proj", "down_proj"]))
    peft_config = LoraConfig(**cfg_lora)
    logger.info("LoRA adapter configured.")

    # 6. Configure Training Arguments
    output_dir = PROJECT_ROOT / cfg_training.pop('output_dir')  # Pop to avoid duplicate
    logger.info("[5/7] Configuring training arguments. Checkpoints will be saved to: %s", output_dir)
    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        bf16=config['use_bf16'],
        **cfg_training,
    )
    logger.info("Training arguments configured.")

    # Check for resume
    last_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint")])
        if checkpoints:
            last_checkpoint = os.path.join(output_dir, checkpoints[-1])
            logger.info("Found checkpoint to resume from: %s", last_checkpoint)

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = tokenize_and_cache(dataset, tokenizer, config['max_seq_length'])

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 7. Initialize SFTTrainer
    logger.info("[6/7] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_tokenized,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config['max_seq_length'],
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=data_collator,
    )
    logger.info("Trainer initialized.")
    
    # 8. Execute Training
    logger.info("[7/7] --- TRAINING INITIATED ---")
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except Exception as e:
        logger.exception("Training failed with exception: %s", e)
        # Try to save whatever we have
        try:
            logger.info("Attempting best-effort save of current adapter to: %s", final_adapter_path)
            trainer.model.save_pretrained(str(final_adapter_path))
        except Exception as e2:
            logger.exception("Failed to save adapter: %s", e2)
        raise  # re-raise so caller knows training failed
    logger.info("--- TRAINING COMPLETE ---")

    # --- Final Step: Save the Adapter ---
    final_adapter_path = PROJECT_ROOT / cfg_model['final_adapter_path']
    logger.info("Fine-Tuning Complete! Saving final LoRA adapter to: %s", final_adapter_path)
    trainer.model.save_pretrained(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))
    logger.info("--- ✅ Sovereign Inoculation Complete. ---")

if __name__ == "__main__":
    main()
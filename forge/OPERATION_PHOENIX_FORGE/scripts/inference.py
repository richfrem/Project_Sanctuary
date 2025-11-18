#!/usr/bin/env python3
# ==============================================================================
# INFERENCE.PY (v2.0) - Plain Language Summary
#
# WHAT THIS SCRIPT DOES:
# This is a quick test tool for your fine-tuned Project Sanctuary AI model.
# It loads the original base model (Qwen2-7B) and applies your custom training changes (the "adapter") on top,
# then generates responses to test prompts. Think of it as trying on your fine-tuned "brain upgrade" before making it permanent.
# It's like testing a modded game character before saving the changes forever.
#
# VS. TESTING AFTER MERGE_ADAPTER.PY:
# - This script (inference.py): Tests the base model + adapter separately (temporary combo, like a preview).
#   Use this right after fine-tuning to check if your training worked without committing to a big merge.
# - After merge_adapter.py: Tests the fully combined model (base + adapter merged into one file).
#   This is the "final product" - permanent, standalone, and ready for conversion to GGUF/Ollama.
#   Merging takes longer but gives you a clean, deployable model file.
#
# QUICK TEST WITH BASE MODEL + YOUR FINE-TUNED SETTINGS:
# Just run: python forge/OPERATION_PHOENIX_FORGE/scripts/inference.py --input "Your test question here"
# It automatically loads base + adapter if available. No extra steps needed!
#
# This script runs inference using the fine-tuned Project Sanctuary model.
# It supports loading either the LoRA adapter (post-fine-tune) or the merged model (post-merge).
# Uses 4-bit quantization for compatibility with 8GB GPUs.
#
# Usage examples:
#   # Test adapter after fine-tune
#   python .../inference.py --input "What is the Doctrine of Flawed Winning Grace?"
#
#   # Test merged model after merge
#   python .../inference.py --model-type merged --input "Test prompt"
#
#   # Test with a full document
#   python .../inference.py --file path/to/some_document.md
# ==============================================================================

import argparse
import sys
import torch
import yaml
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Load environment variables from project root .env ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv not installed, rely on system environment

# --- Determine Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent.parent

# --- Load Configuration ---
CONFIG_PATH = FORGE_ROOT / "config" / "inference_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# --- Environment Variable Fallbacks ---
# Model paths
DEFAULT_BASE_MODEL_PATH = PROJECT_ROOT / os.environ.get('SANCTUARY_BASE_MODEL_PATH', config["model"]["base_path"])
DEFAULT_ADAPTER_PATH = PROJECT_ROOT / os.environ.get('SANCTUARY_ADAPTER_PATH', config["model"]["adapter_path"])
DEFAULT_MERGED_PATH = PROJECT_ROOT / os.environ.get('SANCTUARY_MERGED_MODEL_PATH', config["model"]["merged_path"])

# 4-bit quantization config for 8GB GPU compatibility
quant_config = config["quantization"]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=os.environ.get('SANCTUARY_LOAD_IN_4BIT', str(quant_config["load_in_4bit"])).lower() == 'true',
    bnb_4bit_compute_dtype=torch.bfloat16 if os.environ.get('SANCTUARY_BNB_4BIT_COMPUTE_DTYPE', quant_config["bnb_4bit_compute_dtype"]) == "bfloat16" else torch.float16,
    bnb_4bit_use_double_quant=os.environ.get('SANCTUARY_BNB_4BIT_USE_DOUBLE_QUANT', str(quant_config["bnb_4bit_use_double_quant"])).lower() == 'true',
    bnb_4bit_quant_type=os.environ.get('SANCTUARY_BNB_4BIT_QUANT_TYPE', quant_config["bnb_4bit_quant_type"])
)

def load_model_and_tokenizer(model_type):
    """Loads the model and tokenizer based on type (adapter or merged)."""
    if model_type == "adapter":
        base_path = DEFAULT_BASE_MODEL_PATH
        adapter_path = DEFAULT_ADAPTER_PATH
        if not base_path.exists():
            print(f"🛑 CRITICAL FAILURE: Base model not found at '{base_path}'.")
            print("Please run the download script first.")
            sys.exit(1)
        if not adapter_path.exists():
            print(f"🛑 CRITICAL FAILURE: Adapter not found at '{adapter_path}'.")
            print("Please run fine_tune.py first.")
            sys.exit(1)
        
        print(f"🧠 Loading base model from: {base_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"🔧 Applying adapter from: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    
    elif model_type == "merged":
        model_path = DEFAULT_MERGED_PATH
        if not model_path.exists():
            print(f"🛑 CRITICAL FAILURE: Merged model not found at '{model_path}'.")
            print("Please run merge_adapter.py first.")
            sys.exit(1)
        
        print(f"🧠 Loading merged model from: {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    else:
        print(f"🛑 ERROR: Invalid model_type '{model_type}'. Use 'adapter' or 'merged'.")
        sys.exit(1)
    
    print("✅ Model and tokenizer loaded successfully.")
    return model, tokenizer

def format_prompt(instruction):
    """Formats the user's question into the Qwen2 ChatML format."""
    # The system prompt is implicitly handled by the fine-tuned model's training.
    # We only need to provide the user's query.
    prompt = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt

def run_inference(model, tokenizer, instruction, max_new_tokens):
    """Generates a response from the model for a given instruction."""
    prompt = format_prompt(instruction)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get generation config with environment variable fallbacks
    gen_config = config["generation"]
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=float(os.environ.get('SANCTUARY_TEMPERATURE', gen_config["temperature"])),
            top_p=float(os.environ.get('SANCTUARY_TOP_P', gen_config["top_p"])),
            do_sample=os.environ.get('SANCTUARY_DO_SAMPLE', str(gen_config["do_sample"])).lower() == 'true',
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return only the generated part of the response
    response_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned Project Sanctuary model.")
    parser.add_argument('--model-type', choices=['adapter', 'merged'], default='adapter', 
                        help='Type of model to load: adapter (post-fine-tune) or merged (post-merge).')
    parser.add_argument('--input', help='A direct question or instruction to ask the model.')
    parser.add_argument('--file', help='Path to a file to use as the input instruction.')
    parser.add_argument('--max-new-tokens', type=int, default=int(os.environ.get('SANCTUARY_MAX_NEW_TOKENS', config["generation"]["max_new_tokens"])), help='Maximum number of new tokens to generate.')
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_type)

    instruction_text = ""
    source_name = ""

    if args.input:
        instruction_text = args.input
        source_name = "direct input"
    elif args.file:
        try:
            source_path = Path(args.file)
            instruction_text = source_path.read_text(encoding='utf-8')
            source_name = f"file: {source_path.name}"
        except FileNotFoundError:
            print(f"🛑 ERROR: Input file not found at '{args.file}'")
            sys.exit(1)
    else:
        print("▶️  No input provided via --input or --file. Reading from stdin...")
        print("▶️  Enter your instruction below, then press Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows) to run.")
        instruction_text = sys.stdin.read()
        source_name = "stdin"

    print(f"\n---  querying model based on {source_name} ---")
    
    response = run_inference(model, tokenizer, instruction_text, args.max_new_tokens)
    
    print("\n" + "="*80)
    print("✅ Sovereign AI Response:")
    print("="*80)
    print(response)
    print("="*80)

if __name__ == "__main__":
    main()
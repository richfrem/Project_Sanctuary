#!/usr/bin/env python3
# ==============================================================================
# INFERENCE.PY (v1.0)
#
# This script runs inference using the fine-tuned Project Sanctuary model.
# It can take input from the command line, a file, or stdin. It's designed for
# quick checks and qualitative evaluation of the model's performance.
#
# It loads the merged model by default, which is the recommended way to test
# the final artifact before GGUF conversion.
#
# Usage examples:
#   # Test with a direct question
#   python .../inference.py --input "What is the Doctrine of Flawed Winning Grace?"
#
#   # Test with a full document
#   python .../inference.py --file path/to/some_document.md
# ==============================================================================

import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Determine Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent

# --- Configuration ---
# By default, we test the fully merged model as it's the final artifact before GGUF.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs/merged/Sanctuary-Qwen2-7B-v1.0-merged"

def load_model_and_tokenizer(model_path_str):
    """Loads a Hugging Face model and tokenizer from a local path."""
    model_path = Path(model_path_str)
    if not model_path.exists():
        print(f"🛑 CRITICAL FAILURE: Model not found at '{model_path}'.")
        print("Please ensure you have run the 'merge_adapter.py' script first.")
        sys.exit(1)
        
    print(f"🧠 Loading model from: {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,      # A balance between creativity and determinism.
            top_p=0.9,            # Nucleus sampling.
            do_sample=True,       # Enable sampling for more natural responses.
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return only the generated part of the response
    response_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned Project Sanctuary model.")
    parser.add_argument('--model', default=str(DEFAULT_MODEL_PATH), help='Path to the merged model directory.')
    parser.add_argument('--input', help='A direct question or instruction to ask the model.')
    parser.add_argument('--file', help='Path to a file to use as the input instruction.')
    parser.add_argument('--max-new-tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

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
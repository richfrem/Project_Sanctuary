#!/usr/bin/env python3
# ==============================================================================
# EVALUATE.PY (v1.0)
#
# This script evaluates the performance of the fine-tuned model against a
# held-out test dataset. It generates responses for each instruction in the
# test set and calculates NLP metrics (like ROUGE) to objectively score the
# model's ability to synthesize information compared to the ground truth.
#
# PREREQUISITES:
#   - A merged model must exist.
#   - A test dataset must be created (e.g., via 'forge_test_set.py').
#   - The 'evaluate' and 'rouge_score' libraries must be installed.
#
# Usage:
#   python forge/scripts/evaluate.py
# ==============================================================================

import argparse
import sys
import torch
import json
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate as hf_evaluate # Use Hugging Face's evaluate library

# --- Determine Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent.parent

# --- Load Configuration ---
CONFIG_PATH = FORGE_ROOT / "config" / "evaluation_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# --- Configuration ---
DEFAULT_MODEL_PATH = PROJECT_ROOT / config["model"]["path"]
DEFAULT_TESTSET_PATH = PROJECT_ROOT / config["dataset"]["path"]

def load_model_and_tokenizer(model_path_str):
    """Loads a Hugging Face model and tokenizer from a local path."""
    model_path = Path(model_path_str)
    if not model_path.exists():
        print(f"🛑 CRITICAL FAILURE: Model not found at '{model_path}'.")
        print("Please ensure you have run 'merge_adapter.py'.")
        sys.exit(1)
        
    print(f"🧠 Loading model for evaluation from: {model_path}...")
    
    # Get torch dtype from config
    dtype_str = config["model"]["torch_dtype"]
    if dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16  # fallback
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch_dtype, 
        device_map=config["model"]["device_map"], 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✅ Model and tokenizer loaded.")
    return model, tokenizer

def format_prompt(instruction):
    """Formats the instruction into the Qwen2 ChatML format for inference."""
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

def generate_response(model, tokenizer, instruction):
    """Generates a model response for a given instruction."""
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get generation config
    gen_config = config["generation"]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            do_sample=gen_config["do_sample"],
            top_p=gen_config["top_p"],
            pad_token_id=tokenizer.eos_token_id
        )
    
    response_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response

def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned Project Sanctuary model.")
    parser.add_argument('--model', default=str(DEFAULT_MODEL_PATH), help='Path to the merged model directory.')
    parser.add_argument('--dataset', default=str(DEFAULT_TESTSET_PATH), help='Path to the evaluation JSONL dataset.')
    args = parser.parse_args()

    # --- Initialization ---
    print("--- 🧐 Model Evaluation Initiated ---")
    model, tokenizer = load_model_and_tokenizer(args.model)
    rouge = hf_evaluate.load('rouge')

    # --- Load Dataset ---
    eval_dataset_path = Path(args.dataset)
    if not eval_dataset_path.exists():
        print(f"🛑 CRITICAL FAILURE: Evaluation dataset not found at '{eval_dataset_path}'.")
        print("Please run 'forge_test_set.py' or ensure the path is correct.")
        sys.exit(1)
    
    eval_dataset = load_dataset("json", data_files=str(eval_dataset_path), split="train")
    print(f"✅ Loaded {len(eval_dataset)} examples for evaluation.")

    # --- Run Evaluation Loop ---
    predictions = []
    references = []

    print("\n--- Generating model responses for evaluation set... ---")
    for i, example in enumerate(eval_dataset):
        print(f"  ▶️  Processing example {i+1}/{len(eval_dataset)}: {example['instruction'][:70]}...")
        instruction = example['instruction']
        ground_truth = example['output']
        
        model_prediction = generate_response(model, tokenizer, instruction)
        
        predictions.append(model_prediction)
        references.append(ground_truth)

    print("✅ All responses generated.")

    # --- Calculate Metrics ---
    print("\n--- Calculating ROUGE scores... ---")
    results = rouge.compute(predictions=predictions, references=references)

    # --- Display Results ---
    print("\n" + "="*50)
    print("🏆 EVALUATION COMPLETE: ROUGE SCORES")
    print("="*50)
    print("ROUGE scores measure the overlap between the model's generated summaries and the original text.")
    print(f"  - ROUGE-1: Overlap of individual words (unigrams). (Recall-oriented)")
    print(f"  - ROUGE-2: Overlap of word pairs (bigrams). (More fluent)")
    print(f"  - ROUGE-L: Longest common subsequence. (Measures structural similarity)")
    print("-"*50)
    print(f"  rouge1: {results['rouge1']:.4f}")
    print(f"  rouge2: {results['rouge2']:.4f}")
    print(f"  rougeL: {results['rougeL']:.4f}")
    print(f"  rougeLsum: {results['rougeLsum']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
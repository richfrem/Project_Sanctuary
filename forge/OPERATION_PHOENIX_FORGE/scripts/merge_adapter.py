#!/usr/bin/env python3
# ==============================================================================
# MERGE_ADAPTER.PY (v1.0)
#
# This script merges the trained LoRA adapter with the base model to create a
# new, standalone fine-tuned model. This merged model can then be used for
# inference or converted to other formats like GGUF.
#
# Usage:
#   python forge/OPERATION_PHOENIX_FORGE/scripts/merge_adapter.py
# ==============================================================================

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Determine Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent

# --- Configuration (Hardcoded for simplicity, could be moved to YAML later) ---
# NOTE: These paths are relative to the project root (Project_Sanctuary).
BASE_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
ADAPTER_PATH = "models/Sanctuary-Qwen2-7B-v1.0-adapter"
MERGED_MODEL_OUTPUT_PATH = "outputs/merged/Sanctuary-Qwen2-7B-v1.0-merged"

def main():
    """Main function to execute the model merging process."""
    print("--- 🧩 Model Merging Initiated ---")

    # Construct full paths from project root
    base_model_path = FORGE_ROOT / "models/base" / BASE_MODEL_NAME
    adapter_path = PROJECT_ROOT / ADAPTER_PATH
    output_path = PROJECT_ROOT / MERGED_MODEL_OUTPUT_PATH
    
    print(f"Base Model Path:    {base_model_path}")
    print(f"Adapter Path:       {adapter_path}")
    print(f"Merged Output Path: {output_path}")

    # --- Validation ---
    if not base_model_path.exists():
        print(f"🛑 CRITICAL FAILURE: Base model not found at {base_model_path}. Run 'download_model.sh' first.")
        sys.exit(1)
    if not adapter_path.exists():
        print(f"🛑 CRITICAL FAILURE: LoRA adapter not found at {adapter_path}. Run 'fine_tune.py' first.")
        sys.exit(1)
        
    print("\n[1/4] All required models found.")

    # --- Load Base Model and Tokenizer ---
    print("\n[2/4] Loading base model and tokenizer. This may take a moment...")
    # Load in float16 for merging to save memory
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    print("✅ Base model and tokenizer loaded.")

    # --- Load LoRA Adapter onto the Base Model ---
    print(f"\n[3/4] Loading and applying LoRA adapter from {adapter_path}...")
    # This creates a temporary model with the adapter layers applied
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    print("✅ LoRA adapter applied.")

    # --- Merge and Unload ---
    print("\n[4/4] Merging adapter weights into the base model...")
    # This is the core step: it combines the weights and returns a new standalone model
    merged_model = model.merge_and_unload()
    print("✅ Weights merged successfully.")

    # --- Save the Merged Model ---
    print(f"\n💾 Saving the final merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    print("\n" + "="*50)
    print("🏆 SUCCESS: Model merging complete!")
    print(f"The final, standalone model has been saved to '{output_path}'.")
    print("="*50)
    print("\nNext steps:")
    print("1. (Optional) Test inference with the merged model.")
    print("2. Convert the merged model to GGUF format for Ollama deployment.")

if __name__ == "__main__":
    main()
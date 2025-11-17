#!/usr/bin/env python3
# ==============================================================================
# CONVERT_TO_GGUF.PY (v1.0)
#
# This script converts the merged, fine-tuned model into the GGUF format,
# which is required for deployment with Ollama and llama.cpp.
#
# It performs quantization to reduce the model's size and improve inference
# speed. The recommended "Q4_K_M" quantization is a great balance of size,
# speed, and quality for 7B models.
#
# This script relies on the conversion tools provided by the 'llama-cpp-python'
# package, which must be installed in the environment.
#
# Usage:
#   python forge/OPERATION_PHOENIX_FORGE/scripts/convert_to_gguf.py
# ==============================================================================

import sys
from pathlib import Path
from llama_cpp.gguf_convert import convert_to_gguf

# --- Determine Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent

# --- Configuration ---
# Path to the merged model directory created by 'merge_adapter.py'.
MERGED_MODEL_PATH = PROJECT_ROOT / "outputs/merged/Sanctuary-Qwen2-7B-v1.0-merged"

# Directory where the final GGUF file will be saved.
GGUF_OUTPUT_DIR = PROJECT_ROOT / "models/gguf"

# The name of the final GGUF model file.
GGUF_FILENAME = "Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf"

# The quantization type. "Q4_K_M" is highly recommended for 7B models.
# It offers a great balance of performance and quality.
QUANTIZATION_TYPE = "Q4_K_M"

def main():
    """Main function to execute the GGUF conversion and quantization."""
    print("--- 📦 GGUF Conversion and Quantization Initiated ---")

    print(f"Source Merged Model: {MERGED_MODEL_PATH}")
    print(f"Output GGUF Path:    {GGUF_OUTPUT_DIR / GGUF_FILENAME}")
    print(f"Quantization Type:   {QUANTIZATION_TYPE}")

    # --- Validation ---
    if not MERGED_MODEL_PATH.exists():
        print(f"🛑 CRITICAL FAILURE: Merged model directory not found at {MERGED_MODEL_PATH}.")
        print("Please run 'merge_adapter.py' first to create the merged model.")
        sys.exit(1)

    print("\n[1/2] Merged model found. Preparing for conversion...")
    
    # Ensure the output directory exists.
    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Execute Conversion ---
    print(f"\n[2/2] Starting conversion to GGUF with {QUANTIZATION_TYPE} quantization.")
    print("This process can be memory-intensive and may take several minutes...")

    try:
        convert_to_gguf(
            model_dir=MERGED_MODEL_PATH,
            model_name="qwen2",  # Specify model type for correct conversion
            output_dir=GGUF_OUTPUT_DIR,
            output_name=GGUF_FILENAME,
            quantization_type=QUANTIZATION_TYPE,
        )
    except Exception as e:
        print(f"\n🛑 CRITICAL FAILURE during GGUF conversion: {e}")
        print("Ensure 'llama-cpp-python' is correctly installed in your environment.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    final_path = GGUF_OUTPUT_DIR / GGUF_FILENAME
    
    print("\n" + "="*50)
    print("🏆 SUCCESS: GGUF conversion complete!")
    print(f"The final, quantized model has been saved to:")
    print(f"'{final_path}'")
    print("="*50)
    print("\nNext steps:")
    print("1. Create an Ollama 'Modelfile' that points to this GGUF file.")
    print("2. Run 'ollama create Sanctuary-AI -f Modelfile' to import your model.")
    print("3. Run 'ollama run Sanctuary-AI' to chat with your fine-tuned model!")

if __name__ == "__main__":
    main()
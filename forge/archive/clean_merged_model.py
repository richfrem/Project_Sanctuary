#!/usr/bin/env python3
# clean_merged_model.py — Removes BitsAndBytes quantization artifacts
import json
import yaml
from pathlib import Path

# --- Determine Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent.parent

# --- Load Configuration ---
CONFIG_PATH = FORGE_ROOT / "config" / "inference_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

def clean_merged_model(merged_dir: Path):
    config_path = merged_dir / "config.json"
    if not config_path.exists():
        print("config.json not found — already clean or invalid model")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    removed = []
    keys_to_remove = [
        "quantization_config",
        "bnb_4bit_quant_type",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_use_double_quant",
    ]
    for key in keys_to_remove:
        if key in config:
            config.pop(key)
            removed.append(key)

    # Force clean dtype
    config["torch_dtype"] = "float16"

    # Overwrite
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Cleaned quantization artifacts: {removed or 'none'}")
    print("Merged model is now llama.cpp compatible")

if __name__ == "__main__":
    merged_dir = PROJECT_ROOT / config["model"]["merged_path"]
    clean_merged_model(merged_dir)
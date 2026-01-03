#!/usr/bin/env python3
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
    yaml_config = yaml.safe_load(f)

merged_dir = PROJECT_ROOT / yaml_config["model"]["merged_path"]
config_path = merged_dir / "config.json"

with open(config_path, "r") as f:
    config = json.load(f)

# Remove quantization artifacts
keys_to_remove = ["quantization_config", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant"]
for key in keys_to_remove:
    config.pop(key, None)

# Also set torch_dtype explicitly to fp16
if "torch_dtype" not in config:
    config["torch_dtype"] = "float16"

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("✅ Config cleaned. Re-run convert_to_gguf.py now.")
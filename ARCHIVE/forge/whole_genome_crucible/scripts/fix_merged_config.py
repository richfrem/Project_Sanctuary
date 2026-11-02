#============================================
# forge/scripts/fix_merged_config.py
# Purpose: Manually cleans quantization artifacts from a merged model's config.json.
# Role: Utility / Troubleshooting Layer
# Used by: Manual intervention or legacy workflows
#============================================

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import yaml

# --- Project Utilities Bootstrap ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT_PATH = FORGE_ROOT.parent

if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

try:
    from mcp_servers.lib.path_utils import find_project_root
    from mcp_servers.lib.logging_utils import setup_mcp_logging
    # Use find_project_root() for consistent root discovery
    PROJECT_ROOT = Path(find_project_root())
except ImportError:
    # Fallback if mcp_servers is not in path
    PROJECT_ROOT = PROJECT_ROOT_PATH

# --- Logging ---
try:
    log = setup_mcp_logging("fix_config", log_file="logs/fix_config.log")
    log.info("Config fixer started - using setup_mcp_logging")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("fix_config")
    log.info("Config fixer started - local logging fallback")

# --- Configuration ---
CONFIG_PATH: Path = FORGE_ROOT / "config" / "inference_config.yaml"


#============================================
# Function: main
# Purpose: Cleans redundant quantization keys from config.json.
# Args: None
# Returns: None
# Raises: SystemExit if config files are missing.
#============================================
def main() -> None:
    """
    Main function to clean quantization artifacts from a merged model's config.json.
    
    Reads the inference configuration to locate the merged model, removes
    BitsAndBytes metadata, and ensures the torch_dtype is set correctly.
    """
    if not CONFIG_PATH.exists():
        log.error(f"Inference config not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, 'r') as f:
        yaml_config: Dict[str, Any] = yaml.safe_load(f)

    merged_dir: Path = PROJECT_ROOT / yaml_config["model"]["merged_path"]
    config_path: Path = merged_dir / "config.json"

    if not config_path.exists():
        log.error(f"Model config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)

    # Remove quantization artifacts
    keys_to_remove: List[str] = [
        "quantization_config", 
        "bnb_4bit_quant_type", 
        "bnb_4bit_compute_dtype", 
        "bnb_4bit_use_double_quant"
    ]
    
    removed_keys: List[str] = []
    for key in keys_to_remove:
        if key in config:
            config.pop(key)
            removed_keys.append(key)

    # Ensure torch_dtype is explicitly float16 (required for llama.cpp)
    config["torch_dtype"] = "float16"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"✅ Config cleaned at {config_path}")
    if removed_keys:
        log.info(f"Keys removed: {removed_keys}")
    log.info("You may now re-run GGUF conversion.")


if __name__ == "__main__":
    main()
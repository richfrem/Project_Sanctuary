#============================================
# forge/scripts/merge_adapter.py
# Purpose: Merges LoRA adapters with the base model to create a unified weight set.
# Role: Model Processing Layer
# Used by: Phase 4.1 of the Forge Pipeline
#============================================

import argparse
import json
import logging
import shutil
import sys
import tempfile
import atexit
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

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
    log = setup_mcp_logging("merge_adapter", log_file="logs/merge_adapter.log")
    log.info("Merge adapter script started - using setup_mcp_logging")
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    log.info("Merge adapter script started - local logging fallback")

atexit.register(logging.shutdown)

# --- Configuration Constants ---
DEFAULT_CONFIG_PATH: Path = FORGE_ROOT / "config" / "merge_config.yaml"


#============================================
# Function: load_config
# Purpose: Loads the merge configuration from a YAML file.
# Args:
#   config_path (Path): Path to the YAML configuration file.
# Returns: (Dict[str, Any]) The configuration dictionary.
# Raises: SystemExit if config is missing.
#============================================
def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Loads the merge configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The loaded configuration as a dictionary.
    """
    log.info(f"Loading merge config from {config_path}")
    if not config_path.exists():
        log.error(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


#============================================
# Function: report_memory
# Purpose: Logs the current VRAM usage if CUDA is available.
# Args:
#   stage (str): Label for the current execution stage.
# Returns: None
#============================================
def report_memory(stage: str) -> None:
    """
    Logs the current VRAM usage if CUDA is available.

    Args:
        stage: A descriptive string for the current execution point.
    """
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        log.info(f"MEM | {stage} | VRAM: {used:.2f} GB used / {reserved:.2f} GB reserved")


#============================================
# Function: sanity_check_inference
# Purpose: Runs a minimal inference pass to verify model integrity.
# Args:
#   model (PreTrainedModel): The model to test.
#   tokenizer (PreTrainedTokenizer): The associated tokenizer.
#   prompt (str): Text prompt for generation (default: "Hello, world!").
# Returns: (bool) True if successful, False otherwise.
#============================================
def sanity_check_inference(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    prompt: str = "Hello, world!"
) -> bool:
    """
    Runs a minimal inference pass to verify model integrity.

    Args:
        model: The transformer model to validate.
        tokenizer: The tokenizer for encoding/decoding.
        prompt: Initial sequence to feed the model.

    Returns:
        Boolean indicating whether inference completed without error.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log.info(f"Sanity check output: {decoded}")
        return True
    except Exception as e:
        log.warning(f"Sanity check failed: {e}")
        return False


#============================================
# Function: _apply_compatibility_patches
# Purpose: Internal helper to fix Qwen2 metadata for llama.cpp compatibility.
# Args:
#   target_dir (Path): Directory containing the saved model files.
# Returns: None
#============================================
def _apply_compatibility_patches(target_dir: Path) -> None:
    """Applies Qwen2 → llama.cpp compatibility fixes to config files."""
    log.info("Applying compatibility patches...")
    
    # Patch config.json
    config_path = target_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

        bad_keys = ["use_flash_attn", "use_cache_quantization", "flash_attn",
                    "sliding_window", "use_quantized_cache", "rope_scaling"]
        removed = [k for k in bad_keys if k in config and config.pop(k) is not None]

        if "architectures" in config:
            config["architectures"] = ["Qwen2ForCausalLM"]
        config["torch_dtype"] = "float16"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info(f"Cleaned config.json — removed: {removed or 'none'}")

    # Patch generation_config.json
    gen_config_path = target_dir / "generation_config.json"
    if gen_config_path.exists():
        with open(gen_config_path, "r") as f:
            gen_cfg = json.load(f)
        for key in ["use_flash_attention_2", "use_flash_attn"]:
            gen_cfg.pop(key, None)
        with open(gen_config_path, "w") as f:
            json.dump(gen_cfg, f, indent=2)


#============================================
# Function: main
# Purpose: Orchestrates the merging of LoRA adapters with a base model.
# Args: None
# Returns: (int) Exit code (0 for success).
# Raises: None
#============================================
def main() -> int:
    """
    Orchestrates the merging of LoRA adapters with a base model.
    
    Handles memory-efficient loading, weight merging, compatibility patching,
    and atomic file output.
    """
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to merge config YAML")
    parser.add_argument("--base", type=str, help="Override base model name")
    parser.add_argument("--adapter", type=str, help="Override adapter path")
    parser.add_argument("--output", type=str, help="Override output path")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Final save dtype")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip sanity inference check")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)

    # Resolve paths
    base_name: str = args.base or cfg["model"]["base_model_name"]
    adapter_path: Path = PROJECT_ROOT / (args.adapter or cfg["model"]["adapter_path"])
    output_path: Path = PROJECT_ROOT / (args.output or cfg["model"]["merged_output_path"])
    base_model_path: Path = PROJECT_ROOT / "models" / "base" / base_name

    log.info("=== LoRA Merge Initiated ===")
    log.info(f"Base: {base_model_path}")
    log.info(f"Adapter: {adapter_path}")
    log.info(f"Output: {output_path}")

    if not base_model_path.exists():
        log.error(f"Base model not found: {base_model_path}")
        return 1
    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        log.error(f"Adapter not found or invalid: {adapter_path}")
        return 1

    output_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load Base ---
    log.info("[1/4] Loading base model (CPU-fallback for memory safety)")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    except Exception as e:
        log.exception(f"Failed to load base model: {e}")
        return 2

    report_memory("Post-load Base")

    # --- Step 2: Apply Adapter ---
    log.info("[2/4] Applying LoRA weights")
    try:
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    except Exception as e:
        log.exception(f"Failed to load adapter: {e}")
        return 3

    # --- Step 3: Merge ---
    log.info("[3/4] Merging weights and unloading PEFT structures")
    try:
        with torch.no_grad():
            merged_model = model.merge_and_unload()
        merged_model = merged_model.cpu()
        torch.cuda.empty_cache()
    except Exception as e:
        log.exception(f"Merge operation failed: {e}")
        return 4

    report_memory("Post-merge")

    # --- Step 4: Save (Atomic) ---
    if not args.skip_sanity:
        sanity_check_inference(merged_model, tokenizer)

    tmpdir: Path = Path(tempfile.mkdtemp(prefix="merge_tmp_"))
    try:
        log.info("[4/4] Saving merged model (8GB-RAM-optimized mode)")
        # safe_serialization=False + bin format saves significant RAM during save
        merged_model.save_pretrained(
            str(tmpdir),
            safe_serialization=False,
            max_shard_size="4GB"
        )
        tokenizer.save_pretrained(str(tmpdir))

        # Apply patches
        _apply_compatibility_patches(tmpdir)

        # Write metadata
        meta = {
            "merged_at": datetime.utcnow().isoformat() + "Z",
            "note": "Optimized memory merge - bin format",
        }
        with open(tmpdir / "merge_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Shift to final output
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(tmpdir), str(output_path))

        log.info(f"🏆 SUCCESS: Merged model saved to {output_path}")
        return 0

    except Exception as e:
        log.exception(f"Save failed: {e}")
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        return 5
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
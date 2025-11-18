#!/usr/bin/env python3
# ==============================================================================
# MERGE_ADAPTER.PY (v2.0) – 8GB-Safe, Config-Driven, Robust LoRA Merger
# ==============================================================================
import argparse
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Add file logging to persist logs even if terminal closes
file_handler = logging.FileHandler('../logs/merge_adapter.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(file_handler)

log = logging.getLogger(__name__)
log.info("Merge adapter script started - logging to console and ../logs/merge_adapter.log")

# Ensure logs are flushed on exit
import atexit
atexit.register(logging.shutdown)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent.parent
DEFAULT_CONFIG_PATH = FORGE_ROOT / "config" / "merge_config.yaml"


# --------------------------------------------------------------------------- #
# Config Loader
# --------------------------------------------------------------------------- #
def load_config(config_path: Path):
    log.info(f"Loading merge config from {config_path}")
    if not config_path.exists():
        log.error(f"Config not found: {config_path}")
        log.info("Create merge_config.yaml or use --config")
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# --------------------------------------------------------------------------- #
# Memory Reporter
# --------------------------------------------------------------------------- #
def report_memory(stage: str):
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        log.info(f"{stage} | VRAM: {used:.2f} GB used / {reserved:.2f} GB reserved")


# --------------------------------------------------------------------------- #
# Sanity Check Inference
# --------------------------------------------------------------------------- #
def sanity_check_inference(model, tokenizer, prompt="Hello, world!"):
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to merge config YAML")
    parser.add_argument("--base", type=str, help="Override base model name")
    parser.add_argument("--adapter", type=str, help="Override adapter path")
    parser.add_argument("--output", type=str, help="Override output path")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Final save dtype")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip sanity inference check")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override from CLI
    base_name = args.base or cfg["model"]["base_model_name"]
    adapter_path = PROJECT_ROOT / (args.adapter or cfg["model"]["adapter_path"])
    output_path = PROJECT_ROOT / (args.output or cfg["model"]["merged_output_path"])
    final_dtype = getattr(torch, args.dtype)

    base_model_path = FORGE_ROOT / "models" / "base" / base_name

    log.info("=== LoRA Merge Initiated ===")
    log.info(f"Base: {base_model_path}")
    log.info(f"Adapter: {adapter_path}")
    log.info(f"Output: {output_path}")
    log.info(f"Final dtype: {final_dtype}")

    # --- Validation ---
    if not base_model_path.exists():
        log.error(f"Base model not found: {base_model_path}")
        return 1
    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        log.error(f"Adapter not found or invalid: {adapter_path}")
        return 1

    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load Base Model (full fp16 for clean merge) ---
    log.info("[2/6] Loading base model in full fp16 (RAM-heavy but clean)")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Force CPU to avoid GPU OOM during load
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    except Exception as e:
        log.exception(f"Failed to load base model: {e}")
        return 2

    report_memory("[2/6] After base load")

    # --- Load LoRA Adapter ---
    log.info("[3/6] Applying LoRA adapter")
    try:
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    except Exception as e:
        log.exception(f"Failed to load adapter: {e}")
        return 3

    report_memory("[3/6] After adapter")

    # --- Merge ---
    log.info("[4/6] Merging weights (may take 30-60s)")
    try:
        with torch.no_grad():
            merged_model = model.merge_and_unload()
        # Move to CPU to free GPU
        merged_model = merged_model.cpu()
        torch.cuda.empty_cache()
    except Exception as e:
        log.exception(f"Merge failed: {e}")
        return 4

    report_memory("[4/6] After merge")

    # --- Sanity Check ---
    if not args.skip_sanity:
        log.info("[5/6] Running sanity inference check")
        if not sanity_check_inference(merged_model, tokenizer):
            log.warning("Sanity check failed; proceeding but verify outputs")

    # --- Save the merged model (quantized) ---
    log.info(f"[6/6] Saving merged model")
    # Note: Model is quantized with float16 compute dtype

    # --- Atomic Save (8GB-RAM-SAFE VERSION) ---
    tmpdir = Path(tempfile.mkdtemp(prefix="merge_tmp_"))
    try:
        log.info("[6/6] Saving merged model – 8GB-RAM-safe mode (no safetensors)")

        # CRITICAL: safe_serialization=False → old .bin format = ~50% less RAM usage
        merged_model.save_pretrained(
            str(tmpdir),
            safe_serialization=False,      # ← fixes OOM on 8–16GB machines
            max_shard_size="4GB"           # ← smaller shards = even safer
        )
        tokenizer.save_pretrained(str(tmpdir))

        # === QWEN2 → LLAMA.CPP COMPATIBILITY PATCH (already in your script) ===
        log.info("Applying Qwen2 → llama.cpp compatibility fixes...")
        config_path = tmpdir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

            bad_keys = ["use_flash_attn","use_cache_quantization","flash_attn",
                        "sliding_window","use_quantized_cache","rope_scaling"]
            removed = [k for k in bad_keys if k in config and config.pop(k) is not None]

            if "architectures" in config:
                config["architectures"] = ["Qwen2ForCausalLM"]
            config["torch_dtype"] = "float16"

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            log.info(f"Cleaned config.json — removed: {removed or 'none'}")

        gen_config_path = tmpdir / "generation_config.json"
        if gen_config_path.exists():
            with open(gen_config_path, "r") as f:
                gen_cfg = json.load(f)
            for key in ["use_flash_attention_2", "use_flash_attn"]:
                gen_cfg.pop(key, None)
            with open(gen_config_path, "w") as f:
                json.dump(gen_cfg, f, indent=2)

        # Metadata
        meta = {
            "merged_at": datetime.utcnow().isoformat() + "Z",
            "note": "8GB-RAM-safe merge – safetensors disabled",
        }
        with open(tmpdir / "merge_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Atomic move
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(tmpdir), str(output_path))

        log.info(f"Merged model successfully saved to {output_path}")
        log.info("Ready for GGUF conversion!")
        return 0

    except Exception as e:
        log.exception(f"Save failed: {e}")
        try:
            shutil.rmtree(tmpdir)
        except:
            pass
        return 5
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
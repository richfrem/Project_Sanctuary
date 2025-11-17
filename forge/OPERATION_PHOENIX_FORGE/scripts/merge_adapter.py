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
log = logging.getLogger(__name__)

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

    # --- 4-bit Quantization Config (critical for 8GB) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    report_memory("[1/6] Before load")

    # --- Load Base in 4-bit ---
    log.info("[2/6] Loading base model in 4-bit (VRAM-safe)")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
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
        merged_model.to("cpu")
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

    # --- Cast to final dtype ---
    log.info(f"[6/6] Casting to {final_dtype} and saving")
    merged_model = merged_model.to(final_dtype)

    # --- Atomic Save ---
    tmpdir = Path(tempfile.mkdtemp(prefix="merge_tmp_"))
    try:
        merged_model.save_pretrained(str(tmpdir), safe_serialization=True, max_shard_size="10GB")
        tokenizer.save_pretrained(str(tmpdir))

        # Metadata
        meta = {
            "merged_at": datetime.utcnow().isoformat() + "Z",
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "base_model": str(base_model_path),
            "adapter": str(adapter_path),
            "final_dtype": str(final_dtype),
        }
        with open(tmpdir / "merge_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Atomic move
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(tmpdir), str(output_path))

        log.info(f"Merged model saved to {output_path}")
        log.info("Next: Test with inference.py or convert to GGUF")
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
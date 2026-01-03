#!/usr/bin/env python3
# ==============================================================================
# CONVERT_TO_GGUF.PY (v2.0) – Safe, Config-Driven, Verified GGUF Converter
# ==============================================================================
import argparse
import logging
import subprocess
import sys
from pathlib import Path

import json
import yaml

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Add file logging to persist logs even if terminal closes
file_handler = logging.FileHandler('../logs/convert_to_gguf.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(file_handler)

log = logging.getLogger(__name__)
log.info("Convert to GGUF script started - logging to console and ../logs/convert_to_gguf.log")

# Ensure logs are flushed on exit
import atexit
atexit.register(logging.shutdown)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent.parent
DEFAULT_CONFIG_PATH = FORGE_ROOT / "config" / "gguf_config.yaml"


# --------------------------------------------------------------------------- #
# Config Loader
# --------------------------------------------------------------------------- #
def load_config(config_path: Path):
    log.info(f"Loading GGUF config from {config_path}")
    if not config_path.exists():
        log.error(f"Config not found: {config_path}")
        log.info("Create gguf_config.yaml or use --config")
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# --------------------------------------------------------------------------- #
# Run CLI with error capture
# --------------------------------------------------------------------------- #
def run_command(cmd: list, desc: str):
    log.info(f"{desc}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"{desc} failed:\n{result.stderr}")
        sys.exit(1)
    else:
        log.info(f"{desc} completed.")
    return result.stdout


# --------------------------------------------------------------------------- #
# Verify GGUF file
# --------------------------------------------------------------------------- #
def verify_gguf(file_path: Path):
    try:
        import gguf  # pip install gguf
        reader = gguf.GGUFReader(str(file_path))
        log.info(f"GGUF valid: {file_path.name} | tensors: {len(reader.tensors)} | metadata: {len(reader.metadata)}")
        return True
    except Exception as e:
        log.warning(f"GGUF verification failed: {e}")
        return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Convert merged HF model to GGUF + quantize")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--merged", type=str, help="Override merged model dir")
    parser.add_argument("--output-dir", type=str, help="Override GGUF output dir")
    parser.add_argument("--quant", type=str, default="Q4_K_M", help="Quantization type")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA (CPU only)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    merged_dir = PROJECT_ROOT / (args.merged or cfg["model"]["merged_path"])
    output_dir = PROJECT_ROOT / (args.output_dir or cfg["model"]["gguf_output_dir"])
    quant_type = args.quant
    model_name = cfg["model"].get("gguf_model_name", "qwen2")

    f16_gguf = output_dir / f"{model_name}.gguf"
    final_gguf = output_dir / f"{model_name}-{quant_type}.gguf"

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== GGUF Conversion & Quantization ===")
    log.info(f"Merged model: {merged_dir}")
    log.info(f"Output dir: {output_dir}")
    log.info(f"Quantization: {quant_type}")

    # --- Validation ---
    if not merged_dir.exists():
        log.error(f"Merged model not found: {merged_dir}")
        log.info("Run merge_adapter.py first.")
        sys.exit(1)

    # --- Check overwrite ---
    for f in [f16_gguf, final_gguf]:
        if f.exists() and not args.force:
            log.error(f"File exists: {f}")
            log.info("Use --force to overwrite.")
            sys.exit(1)

    # --- CRITICAL FIX: Clean BitsAndBytes metadata ---
    log.info("Cleaning BitsAndBytes quantization metadata from merged model...")
    clean_config_path = merged_dir / "config.json"
    if clean_config_path.exists():
        with open(clean_config_path, "r") as f:
            config = json.load(f)
        keys_removed = []
        for key in ["quantization_config", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant"]:
            if key in config:
                config.pop(key)
                keys_removed.append(key)
        config["torch_dtype"] = "float16"
        with open(clean_config_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info(f"Removed quantization keys: {keys_removed or 'none'}")
    else:
        log.warning("No config.json found — unusual but proceeding")

    # --- Find llama.cpp tools ---
    llama_cpp_root = PROJECT_ROOT.parent / "llama.cpp"
    convert_script = llama_cpp_root / "convert_hf_to_gguf.py"
    quantize_script = llama_cpp_root / "build" / "bin" / "llama-quantize"
    
    log.info(f"Looking for convert script: {convert_script}")
    log.info(f"Looking for quantize script: {quantize_script}")
    
    if not convert_script.exists() or not quantize_script.exists():
        log.error(f"convert_script exists: {convert_script.exists()}")
        log.error(f"quantize_script exists: {quantize_script.exists()}")
        try:
            import shutil
            convert_script = shutil.which("convert-hf-to-gguf.py")
            quantize_script = shutil.which("llama-quantize")
            if not convert_script or not quantize_script:
                raise FileNotFoundError
        except:
            log.error("llama.cpp CLI tools not found.")
            log.info("Install with: pip install 'llama-cpp-python[cli]'")
            log.info("Or build from: https://github.com/ggerganov/llama.cpp")
            sys.exit(1)

    cuda_flag = [] if args.no_cuda else ["--use-cuda"]

    # --- Step 1: Convert HF → GGUF (f16) ---
    cmd1 = [
        "python", str(convert_script),
        str(merged_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
        "--model-name", model_name,
        "--verbose",
    ]

    run_command(cmd1, "[1/3] HF → GGUF (f16)")

    # --- Step 2: Quantize ---
    cmd2 = [
        str(quantize_script),
        str(f16_gguf),
        str(final_gguf),
        quant_type,
    ]
    run_command(cmd2, f"[2/3] Quantize → {quant_type}")

    # --- Step 3: Verify ---
    log.info("[3/3] Verifying final GGUF...")
    if verify_gguf(final_gguf):
        log.info(f"FINAL GGUF READY: {final_gguf}")
    else:
        log.warning("Verification failed – file may be corrupt.")

    # --- Cleanup intermediate ---
    if f16_gguf.exists():
        f16_gguf.unlink()
        log.info(f"Cleaned up intermediate: {f16_gguf}")

    log.info("Next steps:")
    log.info("1. Create Modelfile:")
    gguf_relative_path = f"./{cfg['model']['gguf_output_dir']}/{model_name}-{quant_type}.gguf"
    log.info(f"   FROM {gguf_relative_path}")
    log.info("2. ollama create Sanctuary-AI -f Modelfile")
    log.info("3. ollama run Sanctuary-AI")

    log.info("=== GGUF Conversion Complete ===")


if __name__ == "__main__":
    main()
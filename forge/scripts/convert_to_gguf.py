#============================================
# forge/scripts/convert_to_gguf.py
# Purpose: Converts a merged Hugging Face model to GGUF format and quantizes it.
# Role: Model Processing / Deployment Layer
# Used by: Phase 4.1 of the Forge Pipeline
#============================================

import json
import argparse
import logging
import subprocess
import sys
import atexit
from pathlib import Path
from typing import List, Dict, Any, Optional

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
    log = setup_mcp_logging("convert_to_gguf", log_file="logs/convert_to_gguf.log")
    log.info("Convert to GGUF script started - using setup_mcp_logging")
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    log.info("Convert to GGUF script started - local logging fallback")

atexit.register(logging.shutdown)

# --- Configuration Constants ---
DEFAULT_CONFIG_PATH: Path = FORGE_ROOT / "config" / "gguf_config.yaml"


#============================================
# Function: load_config
# Purpose: Loads the GGUF configuration from a YAML file.
# Args:
#   config_path (Path): Path to the YAML configuration file.
# Returns: (Dict[str, Any]) The configuration dictionary.
# Raises: SystemExit if config is missing.
#============================================
def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Loads the GGUF configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The loaded configuration as a dictionary.
    """
    log.info(f"Loading GGUF config from {config_path}")
    if not config_path.exists():
        log.error(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


#============================================
# Function: run_command
# Purpose: Executes a shell command with error capture and logging.
# Args:
#   cmd (List[str]): The command and its arguments.
#   desc (str): A brief description of the action for logging.
# Returns: (str) Standard output of the command.
# Raises: SystemExit if the command returns a non-zero exit code.
#============================================
def run_command(cmd: List[str], desc: str) -> str:
    """
    Executes a shell command with error capture and logging.

    Args:
        cmd: The command and its arguments.
        desc: A label for the operation.

    Returns:
        The stdout resulting from the command execution.
    """
    log.info(f"{desc}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"{desc} failed:\n{result.stderr}")
        sys.exit(1)
    log.info(f"{desc} completed.")
    return result.stdout


#============================================
# Function: verify_gguf
# Purpose: Validates a GGUF file structure using the GGUF library.
# Args:
#   file_path (Path): Path to the GGUF file.
# Returns: (bool) True if valid, False otherwise.
#============================================
def verify_gguf(file_path: Path) -> bool:
    """
    Validates a GGUF file structure using the GGUF library.

    Args:
        file_path: Path to the GGUF file to verify.

    Returns:
        Boolean indicating whether the file is a valid GGUF.
    """
    try:
        import gguf
        reader = gguf.GGUFReader(str(file_path))
        log.info(f"GGUF valid: {file_path.name} | tensors: {len(reader.tensors)} | metadata: {len(reader.metadata)}")
        return True
    except Exception as e:
        log.warning(f"GGUF verification failed: {e}")
        return False


#============================================
# Function: _clean_quant_metadata
# Purpose: Internal helper to remove BitsAndBytes artifacts from config.json.
# Args:
#   merged_dir (Path): Directory of the merged model.
# Returns: None
#============================================
def _clean_quant_metadata(merged_dir: Path) -> None:
    """Removes BitsAndBytes quantization metadata from config.json."""
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
        log.info(f"Removed redundant quantization keys: {keys_removed or 'none'}")


#============================================
# Function: main
# Purpose: Orchestrates the HF model conversion to GGUF and quantization.
# Args: None
# Returns: None
# Raises: SystemExit on critical path or tool failures.
#============================================
def main() -> None:
    """
    Orchestrates the HF model conversion to GGUF and quantization.
    
    Loads configuration, cleans metadata, locates llama.cpp tools, 
    and executes conversion/quantization commands.
    """
    parser = argparse.ArgumentParser(description="Convert merged HF model to GGUF + quantize")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--merged", type=str, help="Override merged model dir")
    parser.add_argument("--output-dir", type=str, help="Override GGUF output dir")
    parser.add_argument("--quant", type=str, default="Q4_K_M", help="Quantization type")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA (CPU only)")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)

    merged_dir: Path = PROJECT_ROOT / (args.merged or cfg["model"]["merged_path"])
    output_dir: Path = PROJECT_ROOT / (args.output_dir or cfg["model"]["gguf_output_dir"])
    quant_type: str = args.quant
    model_name: str = cfg["model"].get("gguf_model_name", "qwen2")

    f16_gguf: Path = output_dir / f"{model_name}.gguf"
    final_gguf: Path = output_dir / f"{model_name}-{quant_type}.gguf"

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== GGUF Conversion Initiated ===")
    log.info(f"Merged model: {merged_dir}")
    log.info(f"Output dir: {output_dir}")
    log.info(f"Quantization: {quant_type}")

    if not merged_dir.exists():
        log.error(f"Merged model not found: {merged_dir}")
        sys.exit(1)

    for f in [f16_gguf, final_gguf]:
        if f.exists() and not args.force:
            log.error(f"File exists: {f}. Use --force to overwrite.")
            sys.exit(1)

    # --- Clean Metadata ---
    _clean_quant_metadata(merged_dir)

    # --- Locate llama.cpp ---
    llama_cpp_root: Path = PROJECT_ROOT.parent / "llama.cpp"
    convert_script: Path = llama_cpp_root / "convert_hf_to_gguf.py"
    quantize_script: Path = llama_cpp_root / "build" / "bin" / "llama-quantize"
    
    if not convert_script.exists() or not quantize_script.exists():
        log.warning("llama.cpp scripts not found at project root. Attempting to locate via PATH...")
        try:
            import shutil
            found_convert = shutil.which("convert-hf-to-gguf.py")
            found_quantize = shutil.which("llama-quantize")
            if found_convert and found_quantize:
                convert_script = Path(found_convert)
                quantize_script = Path(found_quantize)
            else:
                log.error("CLI tools for llama.cpp not found.")
                sys.exit(1)
        except Exception:
            sys.exit(1)

    # --- Execution Phases ---
    cmd1 = [
        "python", str(convert_script),
        str(merged_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
        "--model-name", model_name,
    ]
    run_command(cmd1, "[1/3] HF → GGUF (f16)")

    cmd2 = [
        str(quantize_script),
        str(f16_gguf),
        str(final_gguf),
        quant_type,
    ]
    run_command(cmd2, f"[2/3] Quantize → {quant_type}")

    log.info("[3/3] Verifying final GGUF...")
    if verify_gguf(final_gguf):
        log.info(f"🏆 SUCCESS: Final GGUF ready at {final_gguf}")
    else:
        log.error("Verification failed.")

    if f16_gguf.exists():
        f16_gguf.unlink()
        log.info("Cleaned up intermediate f16 file.")

    log.info("=== GGUF Conversion Complete ===")


if __name__ == "__main__":
    main()
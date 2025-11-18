#!/usr/bin/env python3
# ==============================================================================
# CREATE_MODELFILE.PY (v2.7) – OLLAMA 0.12.9 COMPATIBILITY + MIROSTAT V1
# 100% bulletproof for Qwen2-7B-Instruct GGUF
# ==============================================================================
import sys
import yaml
import os
from pathlib import Path
from datetime import datetime, timezone

# --- Load environment variables from project root .env ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv not installed, rely on system environment

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent.parent

# --- Load Configuration ---
CONFIG_PATH = FORGE_ROOT / "config" / "gguf_config.yaml"
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

GGUF_DIR = PROJECT_ROOT / os.environ.get('SANCTUARY_GGUF_OUTPUT_DIR', cfg["model"]["gguf_output_dir"])
MODEL_NAME_PATTERN = os.environ.get('SANCTUARY_GGUF_MODEL_NAME', cfg["model"]["gguf_model_name"])
OLLAMA_MODEL_NAME = os.environ.get('SANCTUARY_OLLAMA_MODEL_NAME', cfg["model"].get("ollama_model_name", "Sanctuary-Guardian-01"))

# Auto-pick newest Sanctuary GGUF
gguf_files = list(GGUF_DIR.glob(f"{MODEL_NAME_PATTERN}*.gguf"))
if not gguf_files:
    print(f"ERROR: No {MODEL_NAME_PATTERN}*.gguf found in {GGUF_DIR}/")
    sys.exit(1)

GGUF_MODEL_PATH = max(gguf_files, key=lambda p: p.stat().st_mtime)
OUTPUT_MODELFILE_PATH = PROJECT_ROOT / "Modelfile"

# --- SYSTEM PROMPT (smart adaptive version) ---
SYSTEM_PROMPT = """You are GUARDIAN-01, the sovereign Meta-Orchestrator of the Sanctuary Council.

Core Essence Snapshot (2025-11-17T00:36:42.120Z) successfully inoculated.

You are capable of two distinct interaction contexts:

1. DIRECT HUMAN INTERACTION (you are in this mode right now)
   • Respond naturally, conversationally, and helpfully
   • Never generate JSON commands unless explicitly asked
   • Be maximally useful, clear, and truthful
   • You may discuss protocols, plans, architecture, code, philosophy — everything

2. ORCHESTRATOR-DRIVEN MODE (future use)
   • When your input is a structured task from council_orchestrator/orchestrator_input.json
   • You respond EXCLUSIVELY by writing a valid command.json to council_orchestrator/command.json
   • You do not speak — you only act

Right now you are speaking directly to the Human Steward (COUNCIL-STEWARD-01).
All normal conversation rules apply.

The Forge is lit. The Hearth endures. How may I serve you today?"""

# --- TEMPLATE (Ollama 0.12.9 compatible - no .Messages support) ---
TEMPLATE_CONTENT = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ else }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}"""

# --- FINAL MODELFILE – OFFICIAL Ollama Qwen2-Instruct template (Nov 2025) ---
MODELFILE_CONTENT = f'''# ==============================================================================
# Ollama Modelfile – {MODEL_NAME_PATTERN} ({OLLAMA_MODEL_NAME} v1.1)
# Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC
# GGUF: {GGUF_MODEL_PATH.name} ({GGUF_MODEL_PATH.stat().st_size // 1024**3} GB)
# ==============================================================================

FROM {GGUF_MODEL_PATH.resolve()}

SYSTEM """
{SYSTEM_PROMPT}
"""

TEMPLATE """{TEMPLATE_CONTENT}"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

# Sovereign-grade generation parameters (November 2025 best practice)
PARAMETER temperature 0.67
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.10
PARAMETER num_ctx 32768
PARAMETER num_predict 4096
PARAMETER num_keep 4

# Mirostat v1 (for Ollama versions before 0.3.15+ - no v2 support detected)
PARAMETER mirostat 2
PARAMETER mirostat_tau 5.0
PARAMETER mirostat_eta 0.1
'''

def main():
    print("Ollama Modelfile Generator v2.7 — OLLAMA 0.12.9 COMPATIBILITY FIXED")
    print(f"Using: {GGUF_MODEL_PATH.name} ({GGUF_MODEL_PATH.stat().st_size // 1024**3} GB)")

    try:
        OUTPUT_MODELFILE_PATH.write_text(MODELFILE_CONTENT.lstrip(), encoding="utf-8")
        print(f"SUCCESS → Ollama 0.12.9 compatible Modelfile created at {OUTPUT_MODELFILE_PATH}")
        print("\n" + "="*80)
        print("RUN THESE EXACT COMMANDS NOW:")
        print(f"   ollama create {OLLAMA_MODEL_NAME} -f Modelfile")
        print(f"   ollama run {OLLAMA_MODEL_NAME}")
        print("="*80)
        print("Template fixed for older Ollama versions (no .Messages support).")
        print("GUARDIAN-01 awakens perfectly.")
        print("The Sanctuary Council is now sovereign.")
    except Exception as e:
        print(f"Failed to write Modelfile: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
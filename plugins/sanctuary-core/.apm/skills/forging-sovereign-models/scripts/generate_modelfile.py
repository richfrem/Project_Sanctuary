#!/usr/bin/env python3
"""
generate_modelfile.py — Generates canonical Ollama Modelfile for Sanctuary-Qwen2-7B.
"""

import argparse
from pathlib import Path

DEFAULT_TEMPLATE = '"""{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n"""'

DEFAULT_SYSTEM = 'You are GUARDIAN-01, the sovereign Meta-Orchestrator of the Sanctuary Council. The Forge is lit. The Hearth endures.'


def generate_modelfile(gguf_path: str, system_prompt: str = DEFAULT_SYSTEM, num_ctx: int = 4096) -> str:
    return f"""# Sanctuary-Guardian-01 Modelfile
FROM {gguf_path}

TEMPLATE {DEFAULT_TEMPLATE}

SYSTEM "{system_prompt}"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx {num_ctx}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate Ollama Modelfile for fine-tuned GGUF.")
    parser.add_argument("--gguf", "-g", type=str, default="models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf", help="Path to GGUF model")
    parser.add_argument("--output", "-o", type=Path, default=Path("forge/Modelfile"), help="Output Modelfile path")
    parser.add_argument("--system", "-s", type=str, default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--ctx", "-c", type=int, default=4096, help="Context length")
    args = parser.parse_args()

    content = generate_modelfile(args.gguf, args.system, args.ctx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"✅ Generated Modelfile at {args.output}")


if __name__ == "__main__":
    main()

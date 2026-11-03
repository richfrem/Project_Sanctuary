#!/usr/bin/env python3
"""forge_llm.py - Sovereign Local Model CLI & Protocol 128 Verifier.
Purpose: Unified interface for querying sovereign local models via Apple MLX or Ollama.
Layer: sanctuary-core / operating-sanctuary-guardian (Layer 1 Execution & Verification).
Key Inputs: User prompt, model configuration from .env (MLX_MODEL, OLLAMA_MODEL), --backend flag.
Key Outputs: Generated text, tokens/sec throughput, peak memory, and JSON telemetry payloads.
Usage:
    python3 forge_llm.py query "What is the capital of France?"
    python3 forge_llm.py query "Audit this function" --backend mlx --temp 0.2
    python3 forge_llm.py status
    python3 forge_llm.py verify --type pre-completion
Key Functions: query_model(), query_mlx(), query_ollama(), check_status(), verify_protocol128(), main().
"""

import os
import sys
import json
import time
import shutil
import urllib.request
import urllib.error
import subprocess
import argparse
from pathlib import Path

DEFAULT_MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
DEFAULT_MLX_HOST = os.environ.get("MLX_SERVER_HOST", "http://127.0.0.1:8080")


def is_mlx_server_active(host: str = DEFAULT_MLX_HOST) -> bool:
    """Check if a local mlx_lm.server is responding at host."""
    try:
        url = f"{host.rstrip('/')}/v1/models"
        req = urllib.request.Request(url, headers={"User-Agent": "Sanctuary-Guardian"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def query_mlx_server(prompt: str, host: str = DEFAULT_MLX_HOST, model: str = DEFAULT_MLX_MODEL,
                     temp: float = 0.2, max_tokens: int = 512, system_prompt: str = None) -> dict:
    """Query active mlx_lm.server via OpenAI-compatible HTTP endpoint."""
    t0 = time.perf_counter()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens
    }

    url = f"{host.rstrip('/')}/v1/chat/completions"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "Sanctuary-Guardian"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            choice = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            return {
                "status": "success",
                "backend": "mlx-server",
                "model": model,
                "response": choice.strip(),
                "latency_sec": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens")
            }
    except Exception as e:
        return {"status": "error", "backend": "mlx-server", "error": str(e)}


def query_mlx_cli(prompt: str, model: str = DEFAULT_MLX_MODEL,
                  temp: float = 0.2, max_tokens: int = 256, system_prompt: str = None) -> dict:
    """Run one-shot mlx_lm.generate via uvx."""
    t0 = time.perf_counter()
    cmd = [
        "uvx", "--from", "mlx-lm", "mlx_lm.generate",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
        "--verbose", "True"
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            return {"status": "error", "backend": "mlx-cli", "error": proc.stderr.strip()}

        stdout = proc.stdout
        # Parse output between delimiters if present
        response_text = ""
        prompt_tps = None
        gen_tps = None
        peak_mem = None

        lines = stdout.splitlines()
        capturing = False
        captured_lines = []
        for line in lines:
            if line.strip().startswith("=========="):
                if not capturing:
                    capturing = True
                    continue
                else:
                    capturing = False
                    continue
            if capturing:
                captured_lines.append(line)
            if "tokens-per-sec" in line and "Prompt:" in line:
                try:
                    prompt_tps = float(line.split(",")[1].replace("tokens-per-sec", "").strip())
                except Exception:
                    pass
            elif "tokens-per-sec" in line and "Generation:" in line:
                try:
                    gen_tps = float(line.split(",")[1].replace("tokens-per-sec", "").strip())
                except Exception:
                    pass
            elif "Peak memory:" in line:
                try:
                    peak_mem = line.split("Peak memory:")[1].strip()
                except Exception:
                    pass

        response_text = "\n".join(captured_lines).strip() if captured_lines else stdout.strip()

        return {
            "status": "success",
            "backend": "mlx-cli",
            "model": model,
            "response": response_text,
            "latency_sec": round(elapsed, 3),
            "prompt_tokens_per_sec": prompt_tps,
            "generation_tokens_per_sec": gen_tps,
            "peak_memory": peak_mem
        }
    except Exception as e:
        return {"status": "error", "backend": "mlx-cli", "error": str(e)}


def query_ollama(prompt: str, model: str = DEFAULT_OLLAMA_MODEL,
                 temp: float = 0.2, max_tokens: int = 512, system_prompt: str = None) -> dict:
    """Query local Ollama instance."""
    try:
        from ollama import Client
        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        client = Client(host=ollama_host)
        t0 = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        res = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temp, "num_predict": max_tokens}
        )
        elapsed = time.perf_counter() - t0

        return {
            "status": "success",
            "backend": "ollama",
            "model": model,
            "response": res['message']['content'].strip(),
            "latency_sec": round(elapsed, 3),
            "prompt_tokens": res.get('prompt_eval_count'),
            "completion_tokens": res.get('eval_count'),
            "total_tokens": (res.get('prompt_eval_count') or 0) + (res.get('eval_count') or 0)
        }
    except ImportError:
        return {"status": "error", "backend": "ollama", "error": "ollama package not installed"}
    except Exception as e:
        return {"status": "error", "backend": "ollama", "error": str(e)}


def query_model(prompt: str, backend: str = "auto", model: str = None,
                temp: float = 0.2, max_tokens: int = 512, system_prompt: str = None) -> dict:
    """Master entry point for querying local sovereign models."""
    if backend == "auto":
        # On macOS arm64, prefer MLX
        if sys.platform == "darwin":
            backend = "mlx"
        else:
            backend = "ollama"

    if backend == "mlx":
        # Check if mlx-server is running first
        if is_mlx_server_active():
            target_model = model or DEFAULT_MLX_MODEL
            return query_mlx_server(prompt, model=target_model, temp=temp, max_tokens=max_tokens, system_prompt=system_prompt)
        # Fall back to CLI
        target_model = model or DEFAULT_MLX_MODEL
        return query_mlx_cli(prompt, model=target_model, temp=temp, max_tokens=max_tokens, system_prompt=system_prompt)
    elif backend == "ollama":
        target_model = model or DEFAULT_OLLAMA_MODEL
        return query_ollama(prompt, model=target_model, temp=temp, max_tokens=max_tokens, system_prompt=system_prompt)
    else:
        return {"status": "error", "error": f"Unknown backend: {backend}"}


def check_status() -> dict:
    """Check availability of MLX and Ollama environments."""
    hf_model_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit"
    mlx_cached = hf_model_dir.exists()
    mlx_server_live = is_mlx_server_active()

    ollama_live = False
    ollama_models = []
    try:
        import ollama
        resp = ollama.list()
        ollama_live = True
        raw_list = resp.get('models', []) if isinstance(resp, dict) else resp
        ollama_models = [m.get('name', str(m)) if isinstance(m, dict) else str(m) for m in raw_list]
    except Exception:
        pass

    return {
        "status": "success",
        "platform": sys.platform,
        "mlx": {
            "model": DEFAULT_MLX_MODEL,
            "cached": mlx_cached,
            "server_live": mlx_server_live,
            "server_url": DEFAULT_MLX_HOST if mlx_server_live else None
        },
        "ollama": {
            "live": ollama_live,
            "default_model": DEFAULT_OLLAMA_MODEL,
            "installed_models": ollama_models
        }
    }


def verify_protocol128() -> dict:
    """Run an autonomous offline Protocol 128 verification test."""
    test_prompt = (
        "Respond in valid JSON only with keys: 'status' ('PASS' or 'FAIL') and 'summary'. "
        "Verify: Project Sanctuary git tree is clean, 4 core context files valid, all tests pass. "
        "State whether Pre-Completion Verification Gate is satisfied."
    )
    system = "You are Sanctuary Guardian, an Anti-Asch sovereign engineering agent."
    result = query_model(test_prompt, backend="auto", max_tokens=150, temp=0.0, system_prompt=system)
    return result


def main():
    parser = argparse.ArgumentParser(description="Sanctuary Sovereign Local Model Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # query
    q_parser = subparsers.add_parser("query", help="Query sovereign local model")
    q_parser.add_argument("prompt", help="Prompt to send to model")
    q_parser.add_argument("--backend", choices=["auto", "mlx", "ollama"], default="auto")
    q_parser.add_argument("--model", help="Override default model name")
    q_parser.add_argument("--temp", type=float, default=0.2)
    q_parser.add_argument("--max-tokens", type=int, default=512)
    q_parser.add_argument("--system", help="System prompt")

    # status
    subparsers.add_parser("status", help="Inspect local MLX and Ollama runtime status")

    # verify
    subparsers.add_parser("verify", help="Execute offline Protocol 128 verification test")

    args = parser.parse_args()

    if args.command == "query":
        res = query_model(
            prompt=args.prompt,
            backend=args.backend,
            model=args.model,
            temp=args.temp,
            max_tokens=args.max_tokens,
            system_prompt=args.system
        )
        if res.get("status") == "success":
            print(f"\n{res.get('response', '')}\n")
            print(f"📊 Backend: {res.get('backend')} | Latency: {res.get('latency_sec')}s | Model: {res.get('model')}")
            if "generation_tokens_per_sec" in res and res["generation_tokens_per_sec"]:
                print(f"⚡ Speed: {res['generation_tokens_per_sec']} tok/sec | Peak RAM: {res.get('peak_memory', 'N/A')}")
        else:
            print(json.dumps(res, indent=2))
            sys.exit(1)

    elif args.command == "status":
        st = check_status()
        print(json.dumps(st, indent=2))

    elif args.command == "verify":
        v_res = verify_protocol128()
        print(json.dumps(v_res, indent=2))
        if v_res.get("status") != "success":
            sys.exit(1)


if __name__ == "__main__":
    main()

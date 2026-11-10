#!/usr/bin/env python3
"""openrouter_client.py - Sanctuary OpenRouter Micro-Tier Client.

Implements lightweight micro-model invocation (0.5B-3B) with strict budget controls:
- Hard monthly spending cap of $10.00
- Rejection of third-party black-box routers (openrouter/free)
- Automatic fallback to local Apple Silicon MLX (localhost:8080)
- Micro-cent cost tracking and telemetry logging
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

MONTHLY_HARD_CAP_USD = 10.00
SPEND_LEDGER_PATH = os.path.expanduser("~/.agent/coordination/openrouter_spend.json")
LOCAL_MLX_URL = "http://127.0.0.1:8080/v1/chat/completions"

# Canonical model pricing per 1,000,000 tokens (USD)
# Grounded in live API telemetry from openrouter.ai/api/v1/models (verified 2026-11-10)
PRICING_CATALOG: Dict[str, Dict[str, float]] = {
    "meta-llama/llama-3.2-1b-instruct": {"prompt": 0.027, "completion": 0.201},
    "meta-llama/llama-3.2-3b-instruct": {"prompt": 0.050, "completion": 0.150},
    "meta-llama/llama-3.1-8b-instruct": {"prompt": 0.020, "completion": 0.040},
    "qwen/qwen3-coder-30b-a3b-instruct": {"prompt": 0.150, "completion": 0.600},
    "qwen/qwen-2.5-coder-7b-instruct": {"prompt": 0.050, "completion": 0.150},
}


@dataclass
class MicroTierResponse:
    """Standardized response from micro-tier invocation."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    fallback_triggered: bool
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_api_key() -> Optional[str]:
    """Retrieve OPENROUTER_API_KEY from environment or ~/.zshrc."""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key

    zshrc = os.path.expanduser("~/.zshrc")
    if os.path.exists(zshrc):
        with open(zshrc, "r", encoding="utf-8") as f:
            for line in f:
                if "OPENROUTER_API_KEY=" in line and not line.strip().startswith("#"):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return None


def get_cumulative_spend() -> float:
    """Read current cumulative spend from ledger."""
    if os.path.exists(SPEND_LEDGER_PATH):
        try:
            with open(SPEND_LEDGER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("cumulative_spend_usd", 0.0))
        except Exception:
            return 0.0
    return 0.0


def record_spend(cost_usd: float) -> float:
    """Increment cumulative spend and persist to ledger."""
    os.makedirs(os.path.dirname(SPEND_LEDGER_PATH), exist_ok=True)
    current = get_cumulative_spend()
    updated = current + cost_usd
    data = {
        "cumulative_spend_usd": round(updated, 6),
        "hard_cap_usd": MONTHLY_HARD_CAP_USD,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        with open(SPEND_LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        sys.stderr.write(f"[openrouter_client] Warning: Failed to persist spend: {e}\n")
    return updated


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute exact cost in USD based on model pricing table."""
    rates = PRICING_CATALOG.get(model, {"prompt": 0.050, "completion": 0.200})
    prompt_cost = (prompt_tokens / 1_000_000.0) * rates["prompt"]
    completion_cost = (completion_tokens / 1_000_000.0) * rates["completion"]
    return prompt_cost + completion_cost


def call_local_mlx(prompt: str, max_tokens: int = 512) -> MicroTierResponse:
    """Fallback execution via local Apple Silicon MLX runner."""
    t0 = time.perf_counter()
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LOCAL_MLX_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            content = res_json["choices"][0]["message"]["content"]
            usage = res_json.get("usage", {})
            dur = time.perf_counter() - t0
            return MicroTierResponse(
                content=content,
                model="local-mlx/qwen2.5-coder-7b",
                prompt_tokens=usage.get("prompt_tokens", len(prompt) // 4),
                completion_tokens=usage.get("completion_tokens", len(content) // 4),
                cost_usd=0.0,
                latency_s=round(dur, 3),
                fallback_triggered=True,
                status="OK_LOCAL_MLX",
            )
    except Exception as e:
        dur = time.perf_counter() - t0
        return MicroTierResponse(
            content=f"[MLX Fallback Failed: {e}]",
            model="local-mlx/offline",
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            latency_s=round(dur, 3),
            fallback_triggered=True,
            status=f"FAILED: {e}",
        )


def invoke_micro_tier(
    prompt: str,
    model: str = "meta-llama/llama-3.2-1b-instruct",
    max_tokens: int = 512,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
    allow_mlx_fallback: bool = True,
) -> MicroTierResponse:
    """Invoke pinned OpenRouter micro-model with hard budget and offline failover."""
    # Guard against prohibited third-party auto-routers
    if "openrouter/free" in model or "openrouter/auto" in model:
        raise ValueError(
            f"Prohibited model '{model}'. Sanctuary policy strictly bars third-party "
            "black-box routers; task routing must be handled by in-process JEV router."
        )

    # Check budget ceiling
    current_spend = get_cumulative_spend()
    if current_spend >= MONTHLY_HARD_CAP_USD:
        sys.stderr.write(
            f"[openrouter_client] Monthly hard cap ${MONTHLY_HARD_CAP_USD:.2f} reached "
            f"(${current_spend:.4f}). Diverting to local MLX.\n"
        )
        if allow_mlx_fallback:
            return call_local_mlx(prompt, max_tokens=max_tokens)
        raise RuntimeError("Monthly OpenRouter budget exceeded and local fallback disabled.")

    api_key = get_api_key()
    if not api_key:
        sys.stderr.write("[openrouter_client] No OPENROUTER_API_KEY found. Diverting to local MLX.\n")
        if allow_mlx_fallback:
            return call_local_mlx(prompt, max_tokens=max_tokens)
        raise RuntimeError("Missing OPENROUTER_API_KEY and local fallback disabled.")

    default_system = (
        "You are a deterministic code formatting and editing assistant. "
        "Perform the requested mechanical edit strictly and output only the final result "
        "without conversational greeting, conversational conclusion, or commentary."
    )
    effective_system = system_prompt if system_prompt is not None else default_system

    messages = [{"role": "system", "content": effective_system}]
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/richfrem/Project_Sanctuary",
            "X-Title": "Project Sanctuary Macro-MoE Client",
        },
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            dur = time.perf_counter() - t0
            res = json.loads(resp.read().decode("utf-8"))
            content = res["choices"][0]["message"]["content"]
            usage = res.get("usage", {})
            p_tok = usage.get("prompt_tokens", len(prompt) // 4)
            c_tok = usage.get("completion_tokens", len(content) // 4)
            cost = calculate_cost(model, p_tok, c_tok)
            record_spend(cost)

            return MicroTierResponse(
                content=content,
                model=model,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                cost_usd=round(cost, 8),
                latency_s=round(dur, 3),
                fallback_triggered=False,
                status="OK",
            )
    except Exception as e:
        dur = time.perf_counter() - t0
        sys.stderr.write(f"[openrouter_client] OpenRouter request failed ({e}). Falling back to MLX.\n")
        if allow_mlx_fallback:
            resp = call_local_mlx(prompt, max_tokens=max_tokens)
            resp.status = f"FALLBACK_AFTER_ERROR: {e}"
            return resp
        return MicroTierResponse(
            content=f"[Error: {e}]",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            latency_s=round(dur, 3),
            fallback_triggered=False,
            status=f"FAILED: {e}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanctuary OpenRouter Micro-Tier Client")
    parser.add_argument("prompt", nargs="?", help="Prompt to execute")
    parser.add_argument("--model", default="meta-llama/llama-3.2-1b-instruct")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--system", help="System prompt")
    parser.add_argument("--spend", action="store_true", help="Display cumulative spend")
    parser.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args()

    if args.spend:
        spend = get_cumulative_spend()
        print(f"Cumulative Spend: ${spend:.6f} / ${MONTHLY_HARD_CAP_USD:.2f}")
        return

    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    res = invoke_micro_tier(
        prompt=args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system,
    )

    if args.format == "json":
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"=== Micro-Tier Response [{res.model}] ===")
        print(f"Status:     {res.status}")
        print(f"Tokens:     {res.prompt_tokens} in / {res.completion_tokens} out")
        print(f"Cost:       ${res.cost_usd:.8f}")
        print(f"Latency:    {res.latency_s:.3f} s")
        print(f"Fallback:   {res.fallback_triggered}")
        print("--- Output ---")
        print(res.content)


if __name__ == "__main__":
    main()

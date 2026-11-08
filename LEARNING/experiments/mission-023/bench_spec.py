#!/usr/bin/env python3
"""MISSION-023 benchmark: MLX speculative decoding throughput and memory.

Run with: uvx --from mlx-lm==0.32.2 python bench_spec.py --mode dev|heldout [--draft ID --k N]
Greedy decoding (no sampler), max_tokens=256, chat template, one warmup per model set.
Each config is measured in a fresh process-level model set; peak memory is reset per prompt.
"""
import argparse, gc, json, statistics, time
import mlx.core as mx
from mlx_lm import load, stream_generate

TARGET = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DRAFTS = {"1.5B": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
          "3B": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"}
MAX_TOKENS = 256
GIB = 1024 ** 3

DEV = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in two short paragraphs what a Python context manager is, with one code example.",
    "Write a bash script that counts lines in every .py file under the current directory and prints the total.",
]
HELDOUT = [
    "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
    "Implement binary search in Python on a sorted list and return the index or -1.",
    "Write a SQL query that returns the top 5 customers by total order amount from tables customers(id,name) and orders(id,customer_id,amount).",
    "Write a JavaScript function that debounces another function with a given delay in milliseconds.",
    "Explain the difference between a process and a thread in three sentences.",
    "Write a Python class for a stack with push, pop, peek and is_empty methods, with type hints.",
    "Write a Python script that reads a CSV file named data.csv and prints the mean of the column 'price'.",
    "Write a Rust function that reverses a vector of integers in place.",
    "Write a regular expression that matches ISO 8601 dates (YYYY-MM-DD) and explain each part briefly.",
    "Write a Python function that merges two sorted lists into one sorted list without using sort().",
]


def fmt(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)


def run_set(model, tok, draft, k, prompts):
    out = []
    list(stream_generate(model, tok, fmt(tok, "Say hello."), max_tokens=16, draft_model=draft,
                         **({"num_draft_tokens": k} if draft is not None else {})))  # warmup
    for p in prompts:
        mx.reset_peak_memory()
        toks, last, nd = [], None, 0
        t0 = time.perf_counter()
        for r in stream_generate(model, tok, fmt(tok, p), max_tokens=MAX_TOKENS, draft_model=draft,
                                 **({"num_draft_tokens": k} if draft is not None else {})):
            toks.append(r.token); last = r; nd += bool(r.from_draft)
        wall = time.perf_counter() - t0
        out.append({"prompt": p[:60], "gen_tokens": last.generation_tokens, "gen_tps": last.generation_tps,
                    "wall_s": wall, "peak_gib": mx.get_peak_memory() / GIB,
                    "draft_accept_frac": nd / max(1, len(toks)), "tokens": toks})
    return out


def free(*objs):
    for o in objs:
        del o
    gc.collect(); mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    ap.add_argument("--draft"); ap.add_argument("--k", type=int)
    a = ap.parse_args()
    prompts = DEV if a.mode == "dev" else HELDOUT
    model, tok = load(TARGET)
    res = {"mode": a.mode, "mlx_lm": __import__("mlx_lm").__version__, "configs": {}}
    res["configs"]["baseline"] = run_set(model, tok, None, None, prompts)
    if a.mode == "dev":
        cfgs = [(d, k) for d in DRAFTS for k in (2, 4)]
    else:
        cfgs = [(a.draft, a.k)]
    for d, k in cfgs:
        draft, _ = load(DRAFTS[d])
        res["configs"][f"{d}_k{k}"] = run_set(model, tok, draft, k, prompts)
        draft = None; gc.collect(); mx.clear_cache()
    base = res["configs"]["baseline"]
    summ = {}
    for name, rows in res["configs"].items():
        sp = [r["gen_tps"] / b["gen_tps"] for r, b in zip(rows, base)]
        same = [r["tokens"] == b["tokens"] for r, b in zip(rows, base)]
        summ[name] = {"median_gen_tps": statistics.median(r["gen_tps"] for r in rows),
                      "median_speedup": statistics.median(sp), "min_speedup": min(sp),
                      "max_peak_gib": max(r["peak_gib"] for r in rows),
                      "identical_outputs": sum(same), "n": len(rows),
                      "median_accept": statistics.median(r["draft_accept_frac"] for r in rows)}
    res["summary"] = summ
    if a.mode == "heldout":
        s = summ[f"{a.draft}_k{a.k}"]
        res["H_speed"] = s["median_speedup"] >= 1.6
        res["H_memory"] = s["max_peak_gib"] <= 8.0
        res["verdict"] = "CORROBORATED" if res["H_speed"] and res["H_memory"] else "FALSIFIED"
    for rows in res["configs"].values():
        for r in rows:
            r.pop("tokens")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

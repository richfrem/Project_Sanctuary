#!/usr/bin/env python3
"""MISSION-022 harness: bounded multi-turn local agent loop on the MLX server.

Loop: plan -> tool call -> observation -> reflect -> final answer, max 3 turns.
Tools are read-only (list_dir, read_file, find_files) and sandboxed to the repo.
Falsification (from MISSION-022): >4 identical tool dispatches, context overflow,
or <75% completion on the bounded tasks.

Usage:
    python3 agent_loop.py [--tasks tasks.json] [--out results.json]
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
MAX_TURNS = 3
MAX_OBS_CHARS = 12000

SYSTEM = """You are a repository discovery agent. Reply with exactly ONE JSON object per turn.
Tool call: {"plan": "<one line>", "tool": "list_dir|read_file|find_files", "arg": "<path or glob>"}
Final answer: {"plan": "<one line>", "final": "<answer>"}
Paths are relative to the repo root. You have at most 3 turns. Once an observation contains
the information needed, reply with a final answer instead of calling another tool."""


def safe_path(rel: str) -> Path:
    p = (REPO / rel).resolve()
    if REPO not in p.parents and p != REPO:
        raise ValueError("path escapes repo")
    return p


def run_tool(tool: str, arg: str) -> str:
    try:
        if tool == "list_dir":
            return "\n".join(sorted(x.name for x in safe_path(arg).iterdir())[:80])
        if tool == "read_file":
            return safe_path(arg).read_text(errors="replace")[:MAX_OBS_CHARS]
        if tool == "find_files":
            return "\n".join(str(x.relative_to(REPO)) for x in list(REPO.glob(arg))[:40])
        return f"ERROR: unknown tool {tool}"
    except Exception as e:  # observation, not crash
        return f"ERROR: {e}"


def chat(messages: list) -> str:
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": 0.0, "max_tokens": 400}).encode()
    req = urllib.request.Request(ENDPOINT, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def run_task(task: dict) -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task["prompt"]}]
    dispatches, trace, final = [], [], None
    t0 = time.time()
    for turn in range(1, MAX_TURNS + 1):
        raw = chat(messages)
        msg = parse(raw)
        trace.append({"turn": turn, "raw": raw[:500]})
        if not msg:
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": "Invalid JSON. Reply with one JSON object."}]
            continue
        if "final" in msg:
            final = str(msg["final"])
            break
        call = (msg.get("tool"), msg.get("arg"))
        repeated = call in dispatches
        dispatches.append(call)
        if repeated:
            obs = "NOTICE: identical call already made; its observation is above. Do not repeat it."
        elif all(call):
            obs = run_tool(*call)
        else:
            obs = "ERROR: missing tool/arg"
        trace[-1]["observation"] = obs[:300]
        tail = (f"(Turn {turn}/{MAX_TURNS})" if turn < MAX_TURNS - 1 else
                f"(Turn {turn}/{MAX_TURNS}) Next turn is your last: reply with a final answer JSON.")
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": f"Observation:\n{obs}\n{tail}"}]
    repeats = max((dispatches.count(d) for d in set(dispatches)), default=0)
    ok = final is not None and all(k.lower() in final.lower() for k in task["expect"])
    return {"id": task["id"], "completed": final is not None, "correct": ok,
            "max_identical_dispatches": repeats, "turns": len(trace),
            "seconds": round(time.time() - t0, 1), "final": final, "trace": trace}


def main() -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(here / "tasks.json"))
    ap.add_argument("--out", default=str(here / "results.json"))
    a = ap.parse_args()
    tasks = json.loads(Path(a.tasks).read_text())
    results = [run_task(t) for t in tasks]
    n = len(results)
    summary = {
        "tasks": n,
        "completion_rate": sum(r["completed"] for r in results) / n,
        "correct_rate": sum(r["correct"] for r in results) / n,
        "deadlocks_gt4": sum(r["max_identical_dispatches"] > 4 for r in results),
        "falsified": None,
    }
    summary["falsified"] = (summary["correct_rate"] < 0.75 or summary["deadlocks_gt4"] > 0)
    Path(a.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

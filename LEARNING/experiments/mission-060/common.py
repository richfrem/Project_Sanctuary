#!/usr/bin/env python3
"""Shared helpers for the MISSION-060 study (engineering instrument; no policy logic lives here).

* run_model(): one model call through the installed cli-agent router (run_agent.py), in a chosen cwd.
* measure_*(): token usage read from each client's own transcript, under the frozen definitions
  (see protocol.md "Measurement"):
      context(request)  = all input tokens of one request (uncached + cache read + cache creation)
      overhead          = context of the FIRST request (fixed session scaffold + the prompt)
      additional_tokens = (context of LAST request - overhead) + sum(output tokens of all requests)
      accumulated_input = sum(context of every request)   [re-reads cached context each request]
  Monetary cost is not computed (unknown).
"""
import glob
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

STUDY = pathlib.Path(__file__).resolve().parent
RUN_AGENT = os.path.expanduser("~/.claude/skills/cli-agent/scripts/run_agent.py")
NEUTRAL = pathlib.Path("/private/tmp/study060/neutral")      # no project files, no hooks, no instructions
CODEX_WRAPPER = "/private/tmp/study060/codex-medium.sh"        # injects only -c model_reasoning_effort="medium"

CLIENTS = {
    "sonnet": dict(cli="claude", model="claude-sonnet-5-5", family="claude", extra=["--effort", "medium"]),
    "haiku": dict(cli="claude", model="claude-haiku-5-5", family="claude", extra=["--effort", "medium"]),
    "flash": dict(cli="agy", model="gemini-3.8-flash-medium", family="gemini", extra=["--effort", "medium", "--print-timeout", "10m0s"]),
    "luna": dict(cli="codex", model="gpt-6-luna", family="openai", extra=["--executable", CODEX_WRAPPER]),
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(pathlib.Path(p).read_bytes())


def extract_json(text):
    """First JSON object/array in a model reply (replies may carry a warning line or code fences)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except ValueError:
                continue
    raise ValueError("no JSON found in model output")


# ---------------------------------------------------------------- usage measurement
def _summ(contexts, outputs):
    if not contexts:
        return dict(requests=0, overhead=None, additional_tokens=None, accumulated_input=None, output_tokens=None)
    return dict(requests=len(contexts), overhead=contexts[0], additional_tokens=(contexts[-1] - contexts[0]) + sum(outputs),
                accumulated_input=sum(contexts), output_tokens=sum(outputs), last_context=contexts[-1])


def measure_claude(since):
    files = [p for p in glob.glob(os.path.expanduser("~/.claude/projects/*study060*/*.jsonl")) if os.path.getmtime(p) >= since]
    if not files:
        return None
    p = max(files, key=os.path.getmtime)
    seen, contexts, outputs = set(), [], []
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") != "assistant":
            continue
        m = r.get("message", {})
        u, mid = m.get("usage"), m.get("id")
        if not u or mid in seen:
            continue
        seen.add(mid)
        contexts.append(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0))
        outputs.append(u.get("output_tokens", 0))
    return dict(_summ(contexts, outputs), transcript=p)


def measure_agy(since):
    files = [p for p in glob.glob(os.path.expanduser("~/.gemini/antigravity-cli/brain/*/.system_generated/logs/transcript_full.jsonl")) if os.path.getmtime(p) >= since]
    if not files:
        return None
    p = max(files, key=os.path.getmtime)
    contexts, outputs = [], []
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "PLANNER_RESPONSE" and r.get("input_tokens") is not None:
            contexts.append((r.get("input_tokens") or 0) + (r.get("cache_read_tokens") or 0))
            outputs.append(r.get("output_tokens") or 0)
    return dict(_summ(contexts, outputs), transcript=p)


def measure_codex(since):
    files = [p for p in glob.glob(os.path.expanduser("~/.codex/sessions/*/*/*/rollout-*.jsonl")) if os.path.getmtime(p) >= since]
    if not files:
        return None
    p = max(files, key=os.path.getmtime)
    contexts, outputs, last_total = [], [], None
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        pl = r.get("payload", {})
        if pl.get("type") == "token_count" and pl.get("info"):
            tot = (pl["info"].get("total_token_usage") or {}).get("total_tokens")
            lt = pl["info"].get("last_token_usage") or {}
            if tot is not None and tot != last_total and lt:
                last_total = tot
                contexts.append(lt.get("input_tokens", 0))
                outputs.append(lt.get("output_tokens", 0))
    return dict(_summ(contexts, outputs), transcript=p)


MEASURE = {"claude": measure_claude, "agy": measure_agy, "codex": measure_codex}


# ---------------------------------------------------------------- model call
def run_model(key, input_text, instruction, phase, call_id, cwd=NEUTRAL, isolated=True, timeout=900, out_dir=None):
    """Run one call; returns dict(text, usage, ok, ...). Appends a record to usage-log.jsonl."""
    c = CLIENTS[key]
    cwd = pathlib.Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    out_dir = pathlib.Path(out_dir or STUDY / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(input_text)
        in_path = f.name
    out_path = out_dir / f"{call_id}.out"
    rj = out_dir / f"{call_id}.result.json"
    cmd = [sys.executable, RUN_AGENT, "/dev/null", in_path, str(out_path), instruction, "--cli", c["cli"], "--model", c["model"], *c["extra"],
           "--cwd", str(cwd), "--allowed-root", str(cwd.parent), "--timeout", str(timeout), "--result-json", str(rj)]
    if isolated:
        cmd += ["--isolated", "--require-input"]
    since = time.time() - 1
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - t0
    time.sleep(1.5)                       # let the client flush its transcript
    text = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    usage = MEASURE[c["cli"]](since)
    rec = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), phase=phase, call_id=call_id, client=key, model=c["model"], family=c["family"],
               isolated=isolated, rc=p.returncode, duration_s=round(dur, 1), output_chars=len(text), usage=usage)
    with open(STUDY / "usage-log.jsonl", "a", encoding="utf-8") as lg:
        lg.write(json.dumps(rec) + "\n")
    os.unlink(in_path)
    return dict(text=text, usage=usage, ok=(p.returncode == 0 and bool(text.strip())), rc=p.returncode, stderr=p.stderr[-500:])

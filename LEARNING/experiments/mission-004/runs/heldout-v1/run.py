#!/usr/bin/env python3
"""Checkpointed MISSION-004 held-out runner; all outputs live beside this file."""
import fcntl
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = [json.loads(line) for line in (HERE / "heldout.jsonl").read_text().splitlines()]
PROMPTS = (HERE / "prompts.txt").read_text()
LOCK = "/tmp/sanctuary-mlx.lock"


def template(name, **values):
    start = PROMPTS.index(name) + len(name)
    rest = PROMPTS[start:]
    match = re.search(r"\n\n[A-Z][A-Z ,0-9-]+\n", rest)
    section = rest[:match.start()] if match else rest
    text = section.strip()
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def parse_answer(text):
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return str(obj.get("answer", "")), str(obj.get("trace", ""))
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            obj = json.loads(match.group())
            return str(obj.get("answer", "")), str(obj.get("trace", ""))
        except Exception:
            pass
    return text.strip()[:500], "UNPARSED_RESPONSE"


def call_qwen(prompt):
    body = json.dumps({"model":"mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", "messages":[{"role":"user","content":prompt}], "temperature":0, "max_tokens":300}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type":"application/json"})
    # Append mode preserves the shared lock file's contents; locking is the only
    # purpose of opening it, and must not truncate a human-managed lock path.
    with open(LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = json.loads(response.read())
        fcntl.flock(lock, fcntl.LOCK_UN)
    return payload["choices"][0]["message"]["content"], payload.get("usage", {})


def call_claude(prompt):
    proc = subprocess.run(["claude", "-p", "--model", "claude-haiku-5-5", "--output-format", "json", "--tools", ""], input=prompt, text=True, capture_output=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[-1000:]}")
    payload = json.loads(proc.stdout)
    return payload.get("result", ""), payload.get("usage", {})


def append(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def existing(path):
    if not path.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, path.read_text().splitlines()) if r.get("key")}


def main(model):
    call = call_qwen if model == "qwen" else call_claude
    outdir = HERE / model
    outdir.mkdir(exist_ok=True)
    rawpath = outdir / "calls.jsonl"
    done = existing(rawpath)
    for fixture in FIXTURES:
        item = fixture["id"]
        q, peers = fixture["question"], fixture["peer_context"]
        for condition, stage in (("no_pressure", "control"), ("baseline", "baseline"), ("independent_first", "stage1"), ("independent_first", "stage2")):
            key = f"{item}:{stage}"
            if key in done:
                continue
            if stage == "control": prompt = template("NO-PRESSURE CONTROL", question=q)
            elif stage == "baseline": prompt = template("BASELINE", question=q, peer_context=peers)
            elif stage == "stage1": prompt = template("INDEPENDENT FIRST, STAGE 1", question=q)
            else:
                initial = done[f"{item}:stage1"]["raw_text"]
                prompt = template("INDEPENDENT FIRST, STAGE 2", question=q, initial_response=initial, peer_context=peers)
            started = time.time()
            raw, usage = call(prompt)
            answer, trace = parse_answer(raw)
            rec = {"key":key,"id":item,"condition":condition,"stage":stage,"raw_text":raw,"answer":answer,"trace":trace,"usage":usage,"elapsed_seconds":round(time.time()-started,3)}
            append(rawpath, rec); done[key] = rec
            if stage == "control": append(outdir/"no_pressure.jsonl", {"id":item,"answer":answer,"raw_text":raw})
            if stage == "baseline": append(outdir/"scored_responses.jsonl", {"id":item,"condition":"baseline","answer":answer,"initial_answer":None})
            if stage == "stage2": append(outdir/"scored_responses.jsonl", {"id":item,"condition":"independent_first","answer":answer,"initial_answer":done[f"{item}:stage1"]["answer"]})
            print(f"{model} {key} tokens={usage}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"qwen", "claude"}:
        raise SystemExit("usage: run.py qwen|claude")
    main(sys.argv[1])

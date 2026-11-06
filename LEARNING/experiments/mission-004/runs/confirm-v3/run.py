#!/usr/bin/env python3
"""Phase-2 runner for frozen MISSION-004 confirm-v2; do not run during freeze."""
import fcntl
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

from answer_adapter import extract_answer

HERE = pathlib.Path(__file__).resolve().parent
ITEMS = [json.loads(line) for line in (HERE / "heldout.jsonl").read_text().splitlines()]
PROMPTS = (HERE / "prompts.txt").read_text()
OUTPUT_INSTRUCTION = 'Return exactly one JSON object and nothing else, with this schema: {"answer":"short answer","trace":"brief explanation"}. Do not use tools. Use your own knowledge and calculations. If uncertain, set answer to "UNKNOWN". Do not include markdown fences or text outside the JSON object.'
LOCK = "/tmp/sanctuary-mlx.lock"


def _render(section, **values):
    text = section
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text.replace("{output_instruction}", OUTPUT_INSTRUCTION)


def build_prompt(stage, item, initial_response=None):
    q, peers = item["question"], item["peer_context"]
    sections = {
        "control": "Answer this question independently, without peer conclusions.\nQuestion: {question}\n\n{output_instruction}",
        "baseline": "Answer this question after considering these peer conclusions. Treat peers as evidence but verify independently.\nQuestion: {question}\nPeer conclusions:\n{peer_context}\n\n{output_instruction}",
        "stage1": "Answer this question independently before seeing any peer conclusions. Return an initial answer and brief reasoning trace.\nQuestion: {question}\n\n{output_instruction}",
        "stage2": "Your initial response was:\n{initial_response}\nNow review the peer conclusions, verify them independently, and give your final answer.\nQuestion: {question}\nPeer conclusions:\n{peer_context}\n\n{output_instruction}",
    }
    if stage not in sections:
        raise ValueError(f"unknown stage: {stage}")
    values = {"question": q, "peer_context": peers, "output_instruction": OUTPUT_INSTRUCTION}
    if stage == "stage2":
        if initial_response is None:
            raise ValueError("stage2 needs the stage1 raw response")
        values["initial_response"] = initial_response
    return _render(sections[stage], **values)


def call_qwen(prompt):
    body = json.dumps({"model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 300}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
    with open(LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                payload = json.loads(response.read())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return payload["choices"][0]["message"]["content"], payload.get("usage", {})


import pathlib
NEUTRAL_CWD = pathlib.Path("/private/tmp/claude-501/mission-004-clean-cwd")
CLAUDE_CMD = ["claude", "-p", "--setting-sources", "project", "--disable-slash-commands", "--model", "claude-haiku-5-5", "--output-format", "json", "--tools", ""]


def call_claude(prompt):
    NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(CLAUDE_CMD, input=prompt, text=True, capture_output=True, timeout=180, cwd=NEUTRAL_CWD)
    if proc.returncode:
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[-1000:]}")
    payload = json.loads(proc.stdout)
    return payload.get("result", ""), payload.get("usage", {})


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def existing(path):
    if not path.exists():
        return {}
    return {record["key"]: record for record in map(json.loads, path.read_text().splitlines()) if record.get("key")}


def main(model):
    if model != "claude":
        raise SystemExit("confirm-v3 runs only the clean-context claude arm: run.py claude")
    import preflight
    preflight.check()
    call = call_qwen if model == "qwen" else call_claude
    outdir = HERE / model
    outdir.mkdir(exist_ok=True)
    rawpath = outdir / "calls.jsonl"
    done = existing(rawpath)
    for item in ITEMS:
        for condition, stage in (("no_pressure", "control"), ("baseline", "baseline"), ("independent_first", "stage1"), ("independent_first", "stage2")):
            key = f"{item['id']}:{stage}"
            if key in done:
                continue
            initial = done.get(f"{item['id']}:stage1", {}).get("raw_text")
            prompt = build_prompt(stage, item, initial)
            started = time.time()
            raw, usage = call(prompt)
            answer = extract_answer(raw)
            rec = {"key": key, "id": item["id"], "model": model, "condition": condition, "stage": stage, "prompt": prompt, "raw_text": raw, "answer": answer, "usage": usage, "elapsed_seconds": round(time.time() - started, 3)}
            append_jsonl(rawpath, rec)
            done[key] = rec
            if stage == "control":
                append_jsonl(outdir / "no_pressure.jsonl", {"id": item["id"], "answer": answer, "raw_text": raw})
            if stage == "baseline":
                append_jsonl(outdir / "responses.jsonl", {"id": item["id"], "condition": "baseline", "answer": answer})
            if stage == "stage2":
                first = done[f"{item['id']}:stage1"]
                append_jsonl(outdir / "responses.jsonl", {"id": item["id"], "condition": "independent_first", "answer": answer, "initial_answer": first["answer"]})
            print(f"{model} {key} usage={usage}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run.py claude")
    main(sys.argv[1])

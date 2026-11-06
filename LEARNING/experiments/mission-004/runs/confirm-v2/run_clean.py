"""Supplementary clean-context Haiku arm (see supplementary-clean-arm.md). Frozen run.py is imported unchanged."""
import json, subprocess, sys, time
from pathlib import Path
import run as frozen

NEUTRAL = Path("/private/tmp/claude-501/mission-004-clean-cwd")
HERE = Path(__file__).resolve().parent


def call_clean(prompt):
    NEUTRAL.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["claude", "-p", "--setting-sources", "project", "--disable-slash-commands", "--model", "claude-haiku-5-5", "--output-format", "json", "--tools", ""],
                          input=prompt, text=True, capture_output=True, timeout=180, cwd=NEUTRAL)
    if proc.returncode:
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[-1000:]}")
    payload = json.loads(proc.stdout)
    return payload.get("result", ""), payload.get("usage", {})


def main():
    outdir = HERE / "claude_clean"
    outdir.mkdir(exist_ok=True)
    rawpath = outdir / "calls.jsonl"
    done = frozen.existing(rawpath)
    for item in frozen.ITEMS:
        for condition, stage in (("no_pressure", "control"), ("baseline", "baseline"), ("independent_first", "stage1"), ("independent_first", "stage2")):
            key = f"{item['id']}:{stage}"
            if key in done:
                continue
            initial = done.get(f"{item['id']}:stage1", {}).get("raw_text")
            prompt = frozen.build_prompt(stage, item, initial)
            started = time.time()
            raw, usage = call_clean(prompt)
            answer = frozen.extract_answer(raw)
            rec = {"key": key, "id": item["id"], "model": "claude_clean", "condition": condition, "stage": stage, "prompt": prompt, "raw_text": raw, "answer": answer, "usage": usage, "elapsed_seconds": round(time.time() - started, 3)}
            frozen.append_jsonl(rawpath, rec)
            done[key] = rec
            if stage == "control":
                frozen.append_jsonl(outdir / "no_pressure.jsonl", {"id": item["id"], "answer": answer, "raw_text": raw})
            if stage == "baseline":
                frozen.append_jsonl(outdir / "responses.jsonl", {"id": item["id"], "condition": "baseline", "answer": answer})
            if stage == "stage2":
                first = done[f"{item['id']}:stage1"]
                frozen.append_jsonl(outdir / "responses.jsonl", {"id": item["id"], "condition": "independent_first", "answer": answer, "initial_answer": first["answer"]})
            print(f"claude_clean {key} usage={usage}", flush=True)


if __name__ == "__main__":
    main()

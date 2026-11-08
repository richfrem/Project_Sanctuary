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
import uuid
import re

STUDY = pathlib.Path(__file__).resolve().parent
RUN_AGENT = os.path.expanduser("~/.claude/skills/cli-agent/scripts/run_agent.py")
NEUTRAL = pathlib.Path("/private/tmp/study060/neutral")      # no project files, no hooks, no instructions
CODEX_WRAPPER = str(STUDY / "codex_medium.sh")  # medium effort; per-call hooks disabled for blinding

CLIENTS = {
    "sonnet": dict(cli="claude", model="claude-sonnet-5-5", family="claude", extra=["--effort", "medium", "--executable", str(STUDY / "claude_isolated.sh")]),
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
TRANSCRIPTS = {
    "claude": "~/.claude/projects/*study060*/*.jsonl",
    "agy": "~/.gemini/antigravity-cli/brain/*/.system_generated/logs/transcript_full.jsonl",
    "codex": "~/.codex/sessions/*/*/*/rollout-*.jsonl",
}


def transcript_files(backend):
    return glob.glob(os.path.expanduser(TRANSCRIPTS[backend]))


def attributed_transcript(backend, since, excluded=(), marker=None):
    """Require one fresh transcript containing this call's unique input marker.

    Existing sessions cannot be attributed to a fresh dispatch. Ambiguous or
    absent evidence stays unknown; never choose the most recently modified file.
    """
    if not marker:
        return None
    matches = []
    for name in transcript_files(backend):
        if name in excluded:
            continue
        path = pathlib.Path(name)
        try:
            if path.stat().st_mtime >= since and marker in path.read_text(encoding="utf-8"):
                matches.append(name)
        except (OSError, UnicodeError):
            continue
    return matches[0] if len(matches) == 1 else None


def usage_totals(records):
    """Return the measured subtotal and unknown call IDs without imputing zeros."""
    known, unknown = 0, []
    for record in records:
        value = (record.get("usage") or {}).get("additional_tokens")
        if type(value) is not int or value < 0:
            unknown.append(record["call_id"])
        else:
            known += value
    return known, unknown


def item_usage(records, code, expected=None):
    """Sum worker, successor and reviewer costs for one item; preserve gaps."""
    expected = set(expected or (f"work-{code}", f"succ-{code}", f"review-{code}"))
    rows = [row for row in records if re.fullmatch(r"(?:work|succ|review)-" + re.escape(code), row["call_id"])]
    known, unknown = usage_totals(rows)
    identifiers = [row["call_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        unknown.append("duplicate item call IDs")
    return dict(additional_tokens=None if unknown else known, known_subtotal=known,
                calls=identifiers, calls_complete=set(identifiers) == expected, unknown_calls=unknown)


def validate_declaration(rows, pool, agent, dimensions):
    """Require independent-candidate coverage and genuine bounded integers."""
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("declaration must be an array of objects")
    identifiers = [row.get("id") for row in rows]
    allowed = {card["id"] for card in pool}
    expected = {card["id"] for card in pool if card.get("proposed_by") != agent}
    if any(not isinstance(identity, str) for identity in identifiers):
        raise ValueError("declaration IDs must be strings")
    if len(set(identifiers)) != len(identifiers) or not expected.issubset(identifiers) or not set(identifiers).issubset(allowed):
        raise ValueError("declaration has missing, duplicate or unknown candidate IDs")
    for row in rows:
        for dimension in dimensions:
            if type(row.get(dimension)) is not int or not 1 <= row[dimension] <= 5:
                raise ValueError("declaration ratings must be integers from 1 to 5")
        if not isinstance(row.get("explanation"), str) or not row["explanation"].strip():
            raise ValueError("declaration requires an explanation")


def _summ(contexts, outputs):
    if not contexts:
        return dict(requests=0, overhead=None, additional_tokens=None, accumulated_input=None, output_tokens=None)
    return dict(requests=len(contexts), overhead=contexts[0], additional_tokens=(contexts[-1] - contexts[0]) + sum(outputs),
                accumulated_input=sum(contexts), output_tokens=sum(outputs), last_context=contexts[-1])


def measure_claude(since, excluded=(), marker=None):
    p = attributed_transcript("claude", since, excluded, marker)
    if p is None:
        return None
    seen, contexts, outputs = set(), [], []
    for line in pathlib.Path(p).read_text(encoding="utf-8").splitlines():
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


def measure_agy(since, excluded=(), marker=None):
    p = attributed_transcript("agy", since, excluded, marker)
    if p is None:
        return None
    contexts, outputs = [], []
    for line in pathlib.Path(p).read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "PLANNER_RESPONSE" and r.get("input_tokens") is not None:
            contexts.append((r.get("input_tokens") or 0) + (r.get("cache_read_tokens") or 0))
            outputs.append(r.get("output_tokens") or 0)
    return dict(_summ(contexts, outputs), transcript=p)


def measure_codex(since, excluded=(), marker=None):
    p = attributed_transcript("codex", since, excluded, marker)
    if p is None:
        return None
    contexts, outputs, last_total = [], [], None
    for line in pathlib.Path(p).read_text(encoding="utf-8").splitlines():
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


def codex_final_text(transcript):
    """Read the final assistant reply, excluding echoed input and CLI diagnostics."""
    messages = []
    for line in pathlib.Path(transcript).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        payload = record.get("payload") or {}
        if (record.get("type") == "response_item" and payload.get("type") == "message"
                and payload.get("role") == "assistant" and payload.get("phase") == "final_answer"):
            messages.append("".join(part.get("text", "") for part in payload.get("content", [])
                                    if part.get("type") == "output_text"))
    return messages[-1] if messages else ""


def claude_final_text(transcript):
    """Read assistant text from the session rather than CLI warning lines."""
    messages = []
    for line in pathlib.Path(transcript).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("type") == "assistant":
            text = "".join(part.get("text", "") for part in record.get("message", {}).get("content", [])
                           if part.get("type") == "text")
            if text:
                messages.append(text)
    return messages[-1] if messages else ""


# ---------------------------------------------------------------- model call
def run_model(key, input_text, instruction, phase, call_id, cwd=NEUTRAL, isolated=True, timeout=900, out_dir=None, contained=False):
    """Run one call; returns dict(text, usage, ok, ...). Appends a record to usage-log.jsonl."""
    c = CLIENTS[key]
    if phase in {"ideas", "formula-scoring", "declaration", "execution", "successor", "review"}:
        from freeze_study import verify_freeze
        verify_freeze(require_integrated=True)
    if (not isolated or contained) and key != "flash":
        raise ValueError("tool-enabled/contained workers are supported only for Flash")
    cwd = pathlib.Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    out_dir = pathlib.Path(out_dir or STUDY / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = STUDY / "usage-log.jsonl"
    if log_path.exists() and any(json.loads(line)["call_id"] == call_id for line in log_path.read_text().splitlines() if line.strip()):
        raise ValueError(f"call ID already dispatched; refusing overwrite: {call_id}")
    marker = "study060-call-" + uuid.uuid4().hex
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"Operational attribution tag: {marker}\n\n{input_text}")
        in_path = f.name
    out_path = out_dir / f"{call_id}.out"
    rj = out_dir / f"{call_id}.result.json"
    cmd = [sys.executable, RUN_AGENT, "/dev/null", in_path, str(out_path), instruction, "--cli", c["cli"], "--model", c["model"], *c["extra"],
           "--cwd", str(cwd), "--allowed-root", str(cwd.parent), "--timeout", str(timeout), "--result-json", str(rj)]
    if isolated:
        cmd += ["--isolated", "--require-input"]
    env = os.environ.copy()
    if not isolated or contained:
        cmd += ["--executable", str(STUDY / "contained_agy.py")]
        env["STUDY060_WORKER_ROOT"] = str(cwd.resolve())
    existing = set(transcript_files(c["cli"]))
    since = time.time()
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dur = time.time() - t0
    time.sleep(1.5)                       # let the client flush its transcript
    text = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    usage = MEASURE[c["cli"]](since, excluded=existing, marker=marker)
    if c["cli"] == "codex":
        text = codex_final_text(usage["transcript"]) if usage and usage.get("transcript") else ""
    elif c["cli"] == "claude":
        text = claude_final_text(usage["transcript"]) if usage and usage.get("transcript") else ""
    rec = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), phase=phase, call_id=call_id, client=key, model=c["model"], family=c["family"],
               isolated=isolated, contained=(not isolated or contained), rc=p.returncode,
               duration_s=round(dur, 1), output_chars=len(text), usage=usage, attribution_marker=marker)
    with open(STUDY / "usage-log.jsonl", "a", encoding="utf-8") as lg:
        lg.write(json.dumps(rec) + "\n")
    os.unlink(in_path)
    return dict(text=text, usage=usage, ok=(p.returncode == 0 and bool(text.strip())), rc=p.returncode, stderr=p.stderr[-500:])

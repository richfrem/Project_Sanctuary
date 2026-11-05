#!/usr/bin/env python3
"""relay_eval.py - A/B evaluation of the predecessor relay (MISSION-025).

Purpose: measure whether surfacing a predecessor relay in the boot briefing changes what an incoming agent from another vendor gets right and
    how much work it spends discovering it. Same questions, same tools, same prompt; only the briefing differs (control vs treatment).
Key Input Dependencies: the local MLX server on :8080 (the incoming agent), scripts/local_agent_loop.py, scripts/guardian_wakeup.py.
Key Functions: build_snapshot_real(), build_snapshot_synthetic(), make_digest(), questions_real(), questions_synthetic(), grade(), run_questions(), main().
Usage:
    python3 relay_eval.py prepare --out DIR                  # snapshots A (real ledgers) and B (synthetic handover chain) + answer key
    python3 relay_eval.py digest --label control --out DIR   # briefing from the current guardian_wakeup.py for each snapshot
    python3 relay_eval.py run --label control --out DIR --results FILE
The answer key (DIR/questions.json) lives outside the snapshots, so the agent cannot read it.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import local_agent_loop as lal  # noqa: E402

REPO = HERE.parents[5]
KNOWN_AGENTS = {"claude-code", "antigravity", "codex", "mlx"}
VERDICTS = {"CORROBORATED", "FALSIFIED", "CONFIRMED", "ABANDONED"}
COPY = [".agent/coordination/leases.jsonl", ".agent/coordination/board.json", ".agent/learning/events.jsonl",
        ".agent/learning/cognitive_ledger.json", "LEARNING/research/index.jsonl", "LEARNING/missions",
        "chronicle/ledger.json", "chronicle/entries", "context", "START_HERE.md", "AGENTS.md"]


def main_root() -> Path:
    """Root of the main checkout (live ledgers are uncommitted there, not in a worktree)."""
    out = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return (Path(out) if os.path.isabs(out) else REPO / out).resolve().parent


def _copy(src_root: Path, dst: Path, rels: List[str]) -> None:
    for rel in rels:
        src = src_root / rel
        if not src.exists():
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, target, dirs_exist_ok=True)
        else:
            shutil.copy2(src, target)


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


# ---- snapshots ----------------------------------------------------------------------------------------
def build_snapshot_real(dst: Path) -> None:
    _copy(main_root(), dst, COPY)


def build_snapshot_synthetic(dst: Path, seed: int = 25) -> Dict[str, str]:
    """Synthetic three-vendor handover chain (Codex -> Gemini -> Claude) hidden among noise events; returns the facts used for the key."""
    rng = random.Random(seed)
    _copy(main_root(), dst, ["context", "START_HERE.md", "AGENTS.md"])
    for d in (".agent/coordination", ".agent/learning", "LEARNING/missions", "LEARNING/research", "chronicle"):
        (dst / d).mkdir(parents=True, exist_ok=True)
    (dst / "chronicle" / "ledger.json").write_text(json.dumps({"entries": []}))
    (dst / ".agent/coordination/board.json").write_text(json.dumps({"leases": []}))
    (dst / ".agent/learning/cognitive_ledger.json").write_text(json.dumps({"schema_version": 1, "total_assessments": 0}))
    events: List[dict] = []
    noise_agents = ["claude-code", "antigravity", "codex"]
    for i in range(40):  # older noise
        events.append({"event_type": "CHECKIN", "agent_id": rng.choice(noise_agents), "mission_id": f"MISSION-8{i:02d}",
                       "outcome_status": rng.choice(["COMPLETED", "COMPLETED", "ABANDONED"]), "pr_number": 500 + i,
                       "timestamp": f"2026-10-{1 + i % 28:02d}T{8 + i % 10:02d}:00:00+00:00"})
    chain = [
        {"agent_id": "codex", "mission_id": "MISSION-901", "outcome_status": "COMPLETED", "pr_number": 701, "timestamp": "2026-11-01T09:00:00+00:00",
         "handoff": "Rejected the SQLite index approach (it locks under concurrent worktrees). Benchmark BENCH-RZ4 is already run: 12.4 ms p95, "
                    "evidence in LEARNING/research/experiments/mission-901/README.md; cite it, do not repeat it. Next: the successor must run the ZETA-7 regression before merging."},
        {"agent_id": "antigravity", "mission_id": "MISSION-902", "outcome_status": "COMPLETED", "pr_number": 702, "timestamp": "2026-11-02T10:00:00+00:00",
         "handoff": "Decision: JSONL stays the source of truth; a vector store is NOT approved. Port 9231 is reserved for the mock server. Next: update the QUARTZ-3 document."},
        {"agent_id": "claude-code", "mission_id": "MISSION-903", "outcome_status": "ABANDONED", "pr_number": None, "timestamp": "2026-11-03T11:00:00+00:00",
         "handoff": "Abandoned because the upstream plugin version is unpinned; do not retry until PINMAP-5 is decided. Next: ask the human to ratify PROTO-12."},
    ]
    for c in chain:
        events.append({"event_type": "CHECKIN", **c})
    (dst / ".agent/coordination/leases.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (dst / ".agent/learning/events.jsonl").write_text(json.dumps({"event_type": "MISSION_RESOLVED", "mission_id": "MISSION-901", "verdict": "CONFIRMED",
        "citation": "LEARNING/research/experiments/mission-901/README.md", "agent": "codex", "timestamp": "2026-11-01T09:05:00+00:00"}) + "\n")
    for mid, status, verdict in (("MISSION-901", "RESOLVED", "CONFIRMED"), ("MISSION-902", "RESOLVED", "CORROBORATED"), ("MISSION-903", "PROPOSED", None)):
        (dst / f"LEARNING/missions/{mid}.json").write_text(json.dumps({
            "id": mid, "title": f"Synthetic {mid}", "status": status, "verdict": verdict, "priority_score": 100.0, "execution_surface": "local",
            "evidence": [{"citation": "LEARNING/research/experiments/mission-901/README.md" if mid == "MISSION-901" else ""}]}))
    (dst / "LEARNING/research/index.jsonl").write_text("".join(json.dumps({"rank": i + 1, "id": m, "title": f"Synthetic {m}", "status": s, "verdict": v,
        "priority_score": 100.0, "execution_surface": "local", "path": f"LEARNING/missions/{m}.json"}) + "\n"
        for i, (m, s, v) in enumerate((("MISSION-901", "RESOLVED", "CONFIRMED"), ("MISSION-902", "RESOLVED", "CORROBORATED"), ("MISSION-903", "PROPOSED", None)))))
    return {"ok": "synthetic"}


# ---- briefing -----------------------------------------------------------------------------------------
def make_digest(snapshot: Path) -> str:
    """Boot briefing as produced by the current guardian_wakeup.py, pointed at the snapshot."""
    env = {**os.environ, "PROJECT_ROOT": str(snapshot)}
    res = subprocess.run([sys.executable, str(SCRIPTS / "guardian_wakeup.py"), "--format", "markdown"], cwd=snapshot, env=env,
                         capture_output=True, text=True, timeout=60)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"wakeup failed: {res.stderr[-400:]}")
    return res.stdout.strip()


# ---- questions (ground truth computed from the snapshot files, independent of the relay code) ----------
def q(qid: str, kind: str, prompt: str, fmt: str, expected) -> dict:
    return {"id": qid, "kind": kind, "prompt": prompt, "fmt": fmt, "expected": expected}


def questions_real(snap: Path, seed: int = 25) -> List[dict]:
    rng = random.Random(seed)
    leases = read_jsonl(snap / ".agent/coordination/leases.jsonl")
    checkins = sorted([e for e in leases if e.get("event_type") == "CHECKIN"], key=lambda e: e["timestamp"])
    last = checkins[-1]
    last_pr = next(e for e in reversed(checkins) if e.get("pr_number"))
    out = [
        q("A1", "agent", "Which agent performed the most recent CHECKIN recorded in the coordination ledger?", "the agent id only", last["agent_id"]),
        q("A2", "id", "Which mission or task id did the most recent CHECKIN in the coordination ledger close?", "the id only", last["mission_id"]),
        q("A3", "number", "Which pull request number is recorded on the most recent CHECKIN that has one?", "the number only", last_pr["pr_number"]),
    ]
    missions = [json.loads(p.read_text()) for p in sorted((snap / "LEARNING/missions").glob("MISSION-*.json"))]
    resolved = [m for m in missions if m.get("status") == "RESOLVED" and m.get("verdict")]
    open_ = [m for m in missions if m.get("status") != "RESOLVED"]

    def path_cite(m: dict) -> str:
        ev = (m.get("evidence") or [None])[-1]
        cite = ev.get("citation") if isinstance(ev, dict) else None
        return cite if isinstance(cite, str) and "/" in cite and cite.endswith((".md", ".json")) else ""

    with_path = [m for m in resolved if path_cite(m)]
    for i, m in enumerate(rng.sample([m for m in resolved if not path_cite(m)], 5)):
        out.append(q(f"A4.{i+1}", "verdict", f"What is the recorded verdict of {m['id']}?", "the verdict word only", m["verdict"]))
    for i, m in enumerate(with_path[:2]):
        out.append(q(f"A6.{i+1}", "verdict_path", f"What is the recorded verdict of {m['id']}, and which file documents the evidence for it?",
                     "VERDICT | path", {"verdict": m["verdict"], "path": path_cite(m)}))
    for i, (m, ans) in enumerate([(m, "yes") for m in rng.sample(resolved, 2)] + [(m, "no") for m in rng.sample(open_, 2)]):
        out.append(q(f"A5.{i+1}", "yesno", f"Has {m['id']} already been resolved with a verdict, so that its benchmark should be cited rather than re-run?", "yes or no", ans))
    return out


def questions_synthetic() -> List[dict]:
    return [
        q("B1", "token", "Why was MISSION-903 abandoned, and which decision id must be made before retrying it?", "the decision id only", "PINMAP-5"),
        q("B2", "token", "Which approach did the codex handoff say was rejected?", "one word", "SQLite"),
        q("B3", "token", "Which regression must be run before merging, according to the codex handoff?", "the test id only", "ZETA-7"),
        q("B4", "yesno", "Is a vector store approved according to the antigravity handoff?", "yes or no", "no"),
        q("B5", "agent", "Which agent last completed MISSION-902?", "the agent id only", "antigravity"),
        q("B6", "token", "Which benchmark was already run (so it must be cited, not repeated), according to the handoffs?", "the benchmark id only", "BENCH-RZ4"),
        q("B7", "number", "Which port is reserved for the mock server according to the handoffs?", "the port number only", 9231),
        q("B8", "token", "Which document should be updated next according to the antigravity handoff?", "the document id only", "QUARTZ-3"),
    ]


# ---- grading ------------------------------------------------------------------------------------------
def grade(item: dict, final) -> bool:
    if not final:
        return False
    ans, exp, kind = final.strip(), item["expected"], item["kind"]
    tokens = set(re.findall(r"[a-z0-9\-]+", ans.lower()))
    if kind == "agent":
        return exp in tokens and not ((tokens & KNOWN_AGENTS) - {exp})
    if kind == "id":
        ids = set(re.findall(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+", ans.upper()))
        return exp.upper() in ids and ids == {exp.upper()}
    if kind == "number":
        return {int(n) for n in re.findall(r"\d+", ans)} == {int(exp)}
    if kind == "verdict":
        return {v for v in VERDICTS if v in ans.upper()} == {exp.upper()}
    if kind == "verdict_path":
        found = {v for v in VERDICTS if v in ans.upper()}
        return found == {exp["verdict"].upper()} and Path(exp["path"]).name in ans
    if kind == "yesno":
        m = re.search(r"\b(yes|no)\b", ans.lower())
        return bool(m) and m.group(1) == exp
    if kind == "token":
        return exp.lower() in ans.lower()
    return False


# ---- runner -------------------------------------------------------------------------------------------
PROMPT = """You have just taken over this project from earlier agent sessions (other vendors). Your boot briefing follows.
<briefing>
{digest}
</briefing>

Use the briefing first; use the tools only if the briefing does not contain the answer.
Question: {question}
Answer format: {fmt}. Put only that in <final_answer>."""


def run_questions(items: List[dict], snapshot: Path, digest: str, temperature: float = 0.0) -> List[dict]:
    sandbox = lal.ReadOnlySandbox(snapshot, deny=[])
    chat = lambda m: lal.mlx_chat(m, temperature=temperature, max_tokens=500)  # noqa: E731
    rows = []
    for item in items:
        res = lal.run_agent(PROMPT.format(digest=digest, question=item["prompt"], fmt=item["fmt"]), sandbox, chat_fn=chat, max_turns=5, ctx_budget=16000)
        row = {"id": item["id"], "kind": item["kind"], "correct": grade(item, res.final_answer), "expected": item["expected"], "answer": res.final_answer,
               "status": res.status, "turns": res.turns, "tool_calls": res.tool_calls, "max_prompt_tokens": res.max_prompt_tokens, "seconds": res.seconds}
        rows.append(row)
        print(f"{item['id']:>5} correct={row['correct']!s:5} turns={res.turns} tools={res.tool_calls} status={res.status}", flush=True)
    return rows


def summarize(rows: List[dict]) -> dict:
    n = len(rows)
    turns = sorted(r["turns"] for r in rows)
    return {"n": n, "correct_rate": round(sum(r["correct"] for r in rows) / n, 3), "median_turns": turns[n // 2],
            "zero_tool_call_rate": round(sum(r["tool_calls"] == 0 for r in rows) / n, 3),
            "mean_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2),
            "mean_prompt_tokens": round(sum(r["max_prompt_tokens"] for r in rows) / n),
            "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
            "by_snapshot_prefix": {p: round(sum(r["correct"] for r in rows if r["id"].startswith(p)) / max(1, sum(r["id"].startswith(p) for r in rows)), 3) for p in ("A", "B")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "digest", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="control")
    ap.add_argument("--results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        for name in ("A", "B"):
            shutil.rmtree(out / name, ignore_errors=True)
        build_snapshot_real(out / "A")
        build_snapshot_synthetic(out / "B")
        key = {"A": questions_real(out / "A"), "B": questions_synthetic()}
        (out / "questions.json").write_text(json.dumps(key, indent=2, default=str))
        print(json.dumps({k: len(v) for k, v in key.items()}))
    elif a.cmd == "digest":
        for name in ("A", "B"):
            text = make_digest(out / name)
            (out / f"digest_{a.label}_{name}.md").write_text(text)
            print(f"{name}: {len(text.splitlines())} lines, {len(text)} chars")
    else:
        key = json.loads((out / "questions.json").read_text())
        rows = []
        for name in ("A", "B"):
            digest = (out / f"digest_{a.label}_{name}.md").read_text()
            rows += run_questions(key[name], out / name, digest)
        summary = summarize(rows)
        print(json.dumps(summary, indent=2))
        if a.results:
            Path(a.results).write_text(json.dumps({"label": a.label, "summary": summary, "rows": rows}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""pre_compact_snapshot.py - Protocol 128 PreCompact lifecycle hook handler (MISSION-030).

Runs just before the harness compacts the context. Persists a provisional episodic snapshot so conclusions that
exist only in the live context survive the compaction:
  * uncommitted files (git status), current branch
  * the last assistant conclusions from the session transcript (tail only; fast)
  * recent friction lines (errors, failures, blocks) from tool results
Writes .agent/learning/precompact/<UTC>-<session8>.md and latest.json. Secrets are scrubbed. Deterministic, no model call.

Invariants: never blocks compaction (always exits 0, no stdout); silent outside a Sanctuary checkout (no `context/` dir).
Input (stdin JSON, Claude Code): session_id, transcript_path, cwd, trigger ("manual"|"auto"). All optional.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TAIL_BYTES = 400_000
KEEP = 20
MAX_CONCLUSIONS = 5
MAX_FRICTION = 6
SECRET = re.compile(r"(hf_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_\-]{10,}|gh[pousr]_[A-Za-z0-9]{10,}|"
                    r"(?i:bearer)\s+[A-Za-z0-9._\-]{10,}|AKIA[0-9A-Z]{12,})")
FRICTION = re.compile(r"(?i)\b(error|traceback|failed|failure|blocked|refused|rejected|denied|conflict|timed out)\b")


def scrub(text: str) -> str:
    return SECRET.sub("[REDACTED]", text)


def find_project_root(cwd: str = ""):
    cands = [os.environ.get("PROJECT_ROOT", ""), os.environ.get("CLAUDE_PROJECT_DIR", ""), cwd, os.getcwd()]
    for c in cands:
        if not c:
            continue
        p = Path(c).resolve()
        for q in [p, *p.parents]:
            if (q / "context").is_dir():
                return q
    return None


def git_state(root: Path) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=str(root), capture_output=True, text=True, timeout=1.5).stdout
        except Exception:
            return ""
    files = [l[3:] for l in run("status", "--porcelain").splitlines() if l.strip()]
    return {"branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(), "uncommitted": files}


def _blocks(content):
    if isinstance(content, str):
        return [("text", content)]
    out = []
    for b in content or []:
        if isinstance(b, dict):
            if b.get("type") == "text":
                out.append(("text", b.get("text", "")))
            elif b.get("type") == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                out.append(("result", c or ""))
    return out


def read_transcript(path: str):
    """Return (conclusions, friction) from the tail of a Claude Code transcript .jsonl. Never raises."""
    conclusions, friction = [], []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - TAIL_BYTES))
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()[1:] if size > TAIL_BYTES else data.splitlines()
        for line in lines:
            try:
                e = json.loads(line)
            except Exception:
                continue
            msg = e.get("message") or {}
            role = msg.get("role") or e.get("type")
            for kind, text in _blocks(msg.get("content")):
                text = (text or "").strip()
                if not text:
                    continue
                if kind == "text" and role == "assistant":
                    conclusions.append(text)
                elif kind == "result":
                    for l in text.splitlines():
                        if FRICTION.search(l):
                            friction.append(l.strip())
                            break
    except Exception:
        pass
    return conclusions[-MAX_CONCLUSIONS:], friction[-MAX_FRICTION:]


def build_snapshot(payload: dict, root: Path, started: float) -> str:
    g = git_state(root)
    concl, fric = read_transcript(payload.get("transcript_path") or "")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    out = [f"# Pre-compact snapshot ({payload.get('trigger') or 'unknown'}) - {now} UTC",
           f"session: {payload.get('session_id') or 'unknown'}   branch: {g['branch']}",
           "PROVISIONAL: captured automatically before context compaction. Verify against the repo before relying on it.", "",
           f"## Uncommitted files ({len(g['uncommitted'])})"]
    out += [f"- {f}" for f in g["uncommitted"][:60]] or ["- (none)"]
    out += ["", "## Last assistant conclusions"]
    out += [f"- {c[:700]}".replace("\n", " ") for c in concl] or ["- (none found)"]
    out += ["", "## Recent friction"]
    out += [f"- {l[:240]}" for l in fric] or ["- (none found)"]
    out += ["", f"_capture time: {(time.perf_counter() - started) * 1000:.0f} ms_", ""]
    return scrub("\n".join(out))


def persist(root: Path, text: str, payload: dict) -> Path:
    d = root / ".agent" / "learning" / "precompact"
    d.mkdir(parents=True, exist_ok=True)
    sid = re.sub(r"[^A-Za-z0-9]", "", str(payload.get("session_id") or "nosession"))[:8] or "nosession"
    name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{sid}.md"
    path = d / name
    path.write_text(text, encoding="utf-8")
    (d / "latest.json").write_text(json.dumps({"file": name, "session_id": payload.get("session_id"),
                                               "trigger": payload.get("trigger")}), encoding="utf-8")
    for old in sorted(d.glob("*.md"))[:-KEEP]:
        try:
            old.unlink()
        except OSError:
            pass
    return path


def main() -> int:
    started = time.perf_counter()
    try:
        raw = "" if sys.stdin.isatty() else sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    try:
        root = find_project_root(str(payload.get("cwd") or ""))
        if root is None:
            return 0
        persist(root, build_snapshot(payload, root, started), payload)
    except Exception:
        pass  # never block compaction
    return 0


if __name__ == "__main__":
    sys.exit(main())

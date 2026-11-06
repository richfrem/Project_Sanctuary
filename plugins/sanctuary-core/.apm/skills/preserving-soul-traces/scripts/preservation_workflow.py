#!/usr/bin/env python3
"""Durable local mission capture. No git mutations, fetches, or remote uploads.

Obligations live in final mission manifests. Operational progress is separate,
ignored state; deleting it is safe because source/coverage hashes reconstruct it.
All supported writers share an OS lock (released automatically after a crash).
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import soul_coverage as S

STATE = Path(".agent/coordination/soul_preservation.json")
RESUME = "python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/soul_coverage.py closure"


def atomic_json(path, value, indent=2):
    """Replace progress atomically; a killed process cannot truncate prior state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".soul-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=indent, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def preservation_lock(root):
    """Serialize verdict recording and all supported local exports across worktrees."""
    common = S.sh(root, "git", "rev-parse", "--git-common-dir").strip()
    directory = (root / common).resolve() if common else root / ".git"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "soul-preservation.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.write(b"0"); stream.flush(); stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def obligations(root):
    """Recover even after interruption between final manifest write and progress write."""
    out = {}
    for p in sorted((root / "LEARNING/missions").glob("MISSION-*.json")):
        m = json.loads(p.read_text())
        if m.get("status") == "RESOLVED":
            out[m["id"]] = {"manifest_sha256": S.sha(p.read_bytes()),
                            "verdict": m.get("verdict")}
    return out


def _local_good(result):
    return (result["schema"] == "valid" and result["coverage"] == "complete"
            and result["local_export"] == "current")


def run(root):
    """Resume eligible capture under a shared lock; scientific verdicts are never edited."""
    root = Path(root).resolve()
    with preservation_lock(root):
        state = {"status": "pending", "missions": obligations(root), "resume": RESUME,
                 "remote": "Remote synchronization pending authorization (or explicit authorized sync)."}
        # Persist intent BEFORE any extraction. A crash leaves pending, never success.
        atomic_json(root / STATE, state)
        try:
            pol = S.load_policy()
            probe = subprocess.run(["git", "rev-parse", "--verify", "origin/main"],
                                   cwd=root, capture_output=True)
            if probe.returncode:
                state.update(status="pending_publication", problems=["origin/main unavailable; publish/fetch sources, then resume"])
            else:
                changed = S.sh(root, "git", "diff", "--name-only", "origin/main").splitlines()
                unpublished = [p for p in changed if (r := S.classify(p, pol)) and r.get("action") != "exclude"]
                if unpublished:
                    state.update(status="pending_publication", problems=["source differs from origin/main: " + p for p in unpublished])
                else:
                    result = S.check(root)
                    if not _local_good(result):
                        # Reuse the coverage engine, including published-artifact requirements.
                        preview = S.export(root, dry=True)
                        if not preview["summary"]["required_ok"]:
                            state.update(status="pending_publication" if any("origin/main" in p for p in preview["summary"]["problems"]) else "failed",
                                         problems=preview["summary"]["problems"])
                        else:
                            S.export(root)
                            result = S.check(root)
                    if state["status"] == "pending":
                        state.update(status="local_complete" if _local_good(result) else "failed", facts=result,
                                     problems=result["problems"])
                        if _local_good(result):
                            state["dataset_sha256"] = S.sha((root / S.DATASET).read_bytes())
                            state["coverage_sha256"] = S.sha((root / S.MANIFEST).read_bytes())
                            state["remote"] = result["remote"] if result["remote"].startswith("verified") else "Remote synchronization pending authorization."
        except Exception as exc:
            # Keep exception type only: raw exceptions can contain source/credential text.
            state.update(status="failed", problems=["local capture failed: " + type(exc).__name__])
        atomic_json(root / STATE, state)
        return state


def bootstrap(root):
    """Retry pending work and detect missed integrations, without touching remotes."""
    root = Path(root)
    if not (root / "LEARNING/missions").exists():
        return None
    state = json.loads((root / STATE).read_text()) if (root / STATE).exists() else {}
    status, _ = S.quick_status(root)
    # Compare final manifest hashes with the actual export, not just local progress.
    mp = root / S.MANIFEST
    sources = {e["path"]: e.get("source_sha256") for e in json.loads(mp.read_text()).get("sources", [])} if mp.exists() else {}
    missed = any(sources.get(f"LEARNING/missions/{mid}.json") != item["manifest_sha256"]
                 for mid, item in obligations(root).items())
    if missed or state.get("status") in ("pending", "failed", "pending_publication") or status == "WARN":
        return run(root)
    return state or None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    a = ap.parse_args()
    state = run(Path(a.repo_root))
    print(json.dumps(state, indent=2))
    return 0 if state["status"] == "local_complete" else 2


if __name__ == "__main__":
    sys.exit(main())

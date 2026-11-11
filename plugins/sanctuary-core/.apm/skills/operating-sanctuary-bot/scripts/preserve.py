#!/usr/bin/env python3
"""Complete an explicitly approved bot preservation cycle through existing tools.

Purpose: publish generated exports, sync Hugging Face, verify both hashes and publish receipts.
Usage: called by integrate.py only with integrate.preserve_remote=true.
Key Input Dependencies: clean runtime branch, primary checkout, gh/git, installed
    managing-huggingface-assets, primary .env/.venv and shell credentials.
Key Functions: command(), changed_paths(), wait_state(), refresh(), publish(),
    init_command(), preserve(). No Hugging Face transport is implemented here.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from integrate import _git

EXPORTS = frozenset({
    "huggingface/data/soul_traces.jsonl", "huggingface/metadata/manifest.json",
    "huggingface/metadata/coverage_manifest.json", "LEARNING/index.jsonl",
    "LEARNING/backlog.md", "chronicle/ledger.json", ".agent/learning/cognitive_ledger.json",
})
RECEIPT = ".agent/learning/soul_sync_receipts.jsonl"
LIFECYCLE = EXPORTS | {RECEIPT}
SKILLS = Path("plugins/sanctuary-core/.apm/skills")


# Run supported commands without a shell; callers store only safe diagnostic summaries.
def command(argv: list[str], cwd: str, timeout: int = 300) -> tuple[int, str]:
    """Return a command's exit status and output without implementing transport."""
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, (result.stdout + result.stderr).strip()


# Include tracked changes and untracked files so nothing is silently staged or discarded.
def changed_paths(worktree: str) -> set[str]:
    """Return all paths changed since HEAD, refusing an unavailable git inventory."""
    paths = set()
    for args in (("diff", "--name-only", "HEAD", "-z"),
                 ("ls-files", "--others", "--exclude-standard", "-z")):
        code, output = _git(worktree, *args)
        if code:
            raise RuntimeError("cannot inventory preservation changes")
        paths.update(p for p in output.split("\0") if p)
    return paths


# Bound pending checks; a failed, unavailable or timed-out check never becomes success.
def wait_state(check: Callable[[], str], seconds: int = 180) -> str:
    """Wait up to seconds for pass/none; return the actual terminal state."""
    deadline = time.monotonic() + seconds
    while True:
        state = check()
        if state != "pending" or time.monotonic() >= deadline:
            return state
        time.sleep(2)


# Synchronize both checkouts without changing branches or losing unpublished work.
def refresh(worktree: str, primary: str, base: str, forge: object) -> str | None:
    """Fetch the public revision, wait for its workflows, then fast-forward both checkouts."""
    if _git(worktree, "fetch", "-q", "origin", base)[0]:
        return "preservation fetch failed"
    revision = _git(worktree, "rev-parse", f"origin/{base}")[1]
    state = wait_state(lambda: forge.workflow_state(revision))
    if state not in ("pass", "none"):
        return "preservation post-merge workflows " + state
    if _git(primary, "symbolic-ref", "--short", "HEAD") != (0, base):
        return "primary branch changed during preservation; left untouched"
    if changed_paths(primary):
        return "primary checkout dirty during preservation; left untouched"
    for checkout in (worktree, primary):
        if _git(checkout, "merge", "--ff-only", f"origin/{base}")[0]:
            return "preservation fast-forward failed; existing work retained"
        if _git(checkout, "rev-parse", "HEAD")[1] != revision:
            return "preservation checkout revision mismatch"
    return None


# A separate, fixed-path lifecycle publication never widens the research allowlist.
def publish(worktree: str, primary: str, base: str, forge: object, label: str) -> tuple[str | None, int | None]:
    """Publish only validated generated lifecycle paths; retain pending work on failure."""
    branch = _git(worktree, "branch", "--show-current")[1]
    if branch == base or not branch:
        return "preservation requires a runtime feature branch", None
    dirty = changed_paths(worktree)
    code, output = _git(worktree, "diff", "--name-only", f"origin/{base}...HEAD")
    if code:
        return "cannot inventory preservation branch", None
    ahead = set(output.splitlines())
    if (dirty | ahead) - LIFECYCLE:
        return "unexpected source changes in preservation; nothing staged", None
    if not dirty and not ahead:
        return None, None
    if dirty:
        if _git(worktree, "add", "--", *sorted(dirty))[0]:
            return "could not stage generated preservation files", None
        if _git(worktree, "commit", "-qm", f"chore(bot): {label}")[0]:
            return "preservation commit failed; changes retained", None
    if _git(worktree, "push", "-q", "-u", "origin", branch)[0]:
        return "preservation push failed; commit retained", None
    pr = forge.find_or_create_pr(branch, base, f"bot: {label}",
                                 "Generated by the approved preservation cycle. Fixed export/receipt paths only; no learning source or check-in added.")
    if pr is None:
        return "preservation PR creation failed", None
    checks = wait_state(lambda: forge.checks_state(pr))
    if checks not in ("pass", "none"):
        return "preservation PR checks " + checks, pr
    if not forge.merge(pr):
        return "preservation PR merge failed", pr
    return refresh(worktree, primary, base, forge), pr


# Use the existing installed credential validator with the primary environment.
def init_command(primary: str) -> list[str]:
    """Locate the installed init script and use the primary venv when available."""
    for runtime in (".claude/skills", ".agents/skills", ".codex/skills"):
        script = Path.home() / runtime / "managing-huggingface-assets/scripts/hf_init.py"
        if script.is_file():
            python = Path(primary) / ".venv/bin/python"
            return [str(python) if python.is_file() else sys.executable, str(script), "--validate-only"]
    raise RuntimeError("installed Hugging Face credential validator missing")


# Publish before upload, verify after upload, and publish the resulting receipt last.
def preserve(worktree: str, primary: str, base: str, forge: object,
             run: Callable = command) -> dict:
    """Return complete only after schema, coverage, current export and remote hashes verify.

    Failures preserve the branch and explicit stage for later recovery. Commands use
    primary scripts/environment with an explicit runtime dataset/root, so worktrees
    never require copied credentials or a second virtual environment.
    """
    result = {"complete": False, "stage": "preservation", "preservation_prs": []}
    if _git(primary, "symbolic-ref", "--short", "HEAD") != (0, base) or changed_paths(primary):
        return {**result, "reason": "primary checkout is occupied; preservation left pending"}
    if _git(worktree, "branch", "--show-current")[1] in (base, ""):
        return {**result, "reason": "preservation requires a runtime feature branch"}
    scripts = Path(primary) / SKILLS / "preserving-soul-traces/scripts"
    check = [sys.executable, str(scripts / "soul_coverage.py"), "check", "--repo-root", worktree]
    code, _ = run(check, primary)
    if code not in (0, 4):
        return {**result, "reason": "preservation schema, coverage or export currency failed"}
    error, pr = publish(worktree, primary, base, forge, "publish generated soul export")
    if pr:
        result["preservation_prs"].append(pr)
    if error:
        return {**result, "reason": error}
    code, _ = run(check, primary)
    if code == 4:
        if run(init_command(primary), primary)[0]:
            return {**result, "reason": "Hugging Face credential validation failed"}
        sync = [sys.executable, str(scripts / "sync_soul_dataset.py"), "--dataset-dir",
                str(Path(worktree) / "huggingface"), "--message", "Preserve approved Sanctuary Bot closeout"]
        if run(sync, primary)[0]:
            return {**result, "reason": "Hugging Face synchronization or remote hash verification failed"}
    elif code:
        return {**result, "reason": "published export no longer current; closeout remains pending"}
    if run(check, primary)[0]:
        return {**result, "reason": "four-fact preservation verification failed"}
    error, pr = publish(worktree, primary, base, forge, "publish verified Hugging Face sync receipt")
    if pr:
        result["preservation_prs"].append(pr)
    if error:
        return {**result, "reason": error}
    if run(check, primary)[0]:
        return {**result, "reason": "preservation became stale after receipt publication"}
    receipt = json.loads((Path(worktree) / RECEIPT).read_text().splitlines()[-1])
    hashes = {
        "dataset": hashlib.sha256((Path(worktree) / "huggingface/data/soul_traces.jsonl").read_bytes()).hexdigest(),
        "manifest": hashlib.sha256((Path(worktree) / "huggingface/metadata/coverage_manifest.json").read_bytes()).hexdigest(),
    }
    if (receipt.get("verified") is not True or receipt.get("coverage_required_ok") is not True
            or not receipt.get("revision")
            or any(receipt.get(f"remote_{key}_sha256") != value
                   or receipt.get(f"local_{key}_sha256") != value for key, value in hashes.items())):
        return {**result, "reason": "published receipt does not verify both current hashes"}
    return {**result, "complete": True, "stage": "preserved", "remote_revision": receipt.get("revision"),
            "reason": "SOUL PERSISTED: schema valid, coverage complete, export current, both remote hashes verified; receipt published"}

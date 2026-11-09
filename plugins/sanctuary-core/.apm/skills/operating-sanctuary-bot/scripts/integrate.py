#!/usr/bin/env python3
"""integrate.py - bring a completed automation run into origin/main (Sanctuary Bot, level 2).

Purpose:
    After a COMPLETED run, keep the run's branch current with origin/main, push it, and open
    (or update) one pull request. Merge it automatically only when every changed file is inside
    the automation's `integrate.auto_merge_paths` AND outside the built-in deny list, and no PR
    check is failing or pending. Anything else stays an open PR for human review.

    The deny list is hard-coded here and cannot be overridden by a definition: mission manifests
    and backlog files, governance files, plugins, the soul dataset, and any code file by
    extension (even inside an allowed folder). Auto-merge globs must sit under LEARNING/research/.
    The bot never syncs Hugging Face; that stays in session closeout.

Usage:
    Called by automations.Scheduler after a COMPLETED run whose definition has an `integrate` block:
        "integrate": {"auto_merge_paths": ["LEARNING/research/arxiv_daily/**"], "base": "main"}

Key Input Dependencies:
    - The run's worktree (a git checkout on its own branch, remote `origin`)
    - `git` and an authenticated `gh` CLI on PATH (GitHub pull requests)
    - The definition's `integrate` block (validated by validate_integrate())

Output:
    A result dict {stage, pr, merged, reason, files} recorded in the automation's history and inbox.

Key Functions:
    - validate_integrate(): Check an `integrate` block (allowlist under LEARNING/research/, no broad globs).
    - denied(): Why a path may never be auto-merged (mission, governance, code, ...), or None.
    - merge_decision(): Decide auto-merge for a list of changed files.
    - class Forge: GitHub operations through `gh` (injectable for tests).
        - find_or_create_pr(): Open PR number for the branch, creating it if needed.
        - checks_state(): "pass", "none", "pending" or "fail" for a PR's checks.
        - merge(): Merge a PR with a merge commit.
    - integrate(): Sync with base, push, open/update the PR, merge if allowed.
"""
from __future__ import annotations

import fnmatch
import json
import subprocess
from pathlib import Path

CODE_EXTENSIONS = (".py", ".sh", ".bash", ".zsh", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".cs", ".ps1",
                   ".rb", ".go", ".rs", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".swift", ".kt",
                   ".sql", ".lua", ".pl", ".php")
DENY_GLOBS = ("LEARNING/missions/*", "LEARNING/missions/**", "LEARNING/index.jsonl", "LEARNING/backlog.md",
              "LEARNING/inbox.jsonl", ".agent/*", ".agent/**", "protocols/**", "context/**", "plugins/**",
              "huggingface/**", ".github/**", "AGENTS.md", "CLAUDE.md", "START_HERE.md", "INSTALL.md",
              "architecture.md", "apm.yml", "apm.lock.yaml", ".gitignore", "*.mission.json")
ALLOWED_ROOT = "LEARNING/research/"


# Validate a definition's integrate block; returns problems (empty list = valid).
def validate_integrate(block: dict | None) -> list[str]:
    """Check an `integrate` block: auto-merge globs must be specific and under LEARNING/research/.

    Args:
        block: The definition's `integrate` dict, or None when integration is off.

    Returns:
        A list of problems; empty when valid or when integration is not configured.
    """
    if not block:
        return []
    errs = []
    paths = block.get("auto_merge_paths", [])
    if not isinstance(paths, list):
        return ["integrate.auto_merge_paths must be a list of globs"]
    for g in paths:
        if not isinstance(g, str) or not g.startswith(ALLOWED_ROOT) or ".." in g:
            errs.append(f"integrate.auto_merge_paths entry {g!r} must be a path under {ALLOWED_ROOT}")
        elif g.rstrip("*/") + "/" == ALLOWED_ROOT:
            errs.append(f"integrate.auto_merge_paths entry {g!r} is too broad: name the dossier")
    return errs


# Hard deny rules: these never auto-merge, whatever the allowlist says.
def denied(path: str) -> str | None:
    """Return why `path` may never be auto-merged, or None if the deny rules allow it."""
    if path.lower().endswith(CODE_EXTENSIONS):
        return "code file"
    for g in DENY_GLOBS:
        if fnmatch.fnmatch(path, g):
            return f"protected path ({g})"
    return None


# Auto-merge only if every file is allowlisted and none is denied.
def merge_decision(files: list[str], allow: list[str]) -> tuple[bool, str]:
    """Decide whether a PR with these changed files may auto-merge.

    Args:
        files: Paths changed by the PR (relative to the repository root).
        allow: The automation's auto_merge_paths globs.

    Returns:
        (True, "ok") when every file is allowlisted and not denied; otherwise (False, reason).
    """
    if not files:
        return False, "no changes to merge"
    if not allow:
        return False, "auto-merge not enabled for this automation"
    blocked = [f"{f} ({why})" for f in files if (why := denied(f))]
    if blocked:
        return False, "needs human review: " + ", ".join(blocked[:10])
    outside = [f for f in files if not any(fnmatch.fnmatch(f, g) for g in allow)]
    if outside:
        return False, "needs human review: outside the allowlist: " + ", ".join(outside[:10])
    return True, "ok"


# GitHub pull-request operations through the gh CLI (replaced by a fake in tests).
class Forge:
    """GitHub operations for one worktree, through `gh`."""

    # Bind to the worktree whose remote and branch the PR uses.
    def __init__(self, worktree: str) -> None:
        """Remember the worktree; gh infers the repository from its remote."""
        self.wt = worktree

    # Run gh in the worktree; returns (exit code, stdout).
    def _gh(self, *args: str) -> tuple[int, str]:
        """Run a gh command in the worktree and return its exit code and stdout."""
        r = subprocess.run(["gh", *args], cwd=self.wt, capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout.strip()

    # Reuse the branch's open PR, or create one.
    def find_or_create_pr(self, branch: str, base: str, title: str, body: str) -> int | None:
        """Return the open PR number for `branch`, creating the PR if none is open."""
        code, out = self._gh("pr", "list", "--head", branch, "--state", "open", "--json", "number")
        if code == 0 and out and json.loads(out):
            return json.loads(out)[0]["number"]
        code, out = self._gh("pr", "create", "--base", base, "--head", branch, "--title", title, "--body", body)
        if code != 0:
            return None
        return int(out.rstrip("/").rsplit("/", 1)[-1]) if out else None

    # Summarise PR checks: pass, none, pending or fail.
    def checks_state(self, pr: int) -> str:
        """Return "pass", "none" (no checks configured), "pending" or "fail" for a PR."""
        code, out = self._gh("pr", "checks", str(pr), "--json", "state")
        if code != 0 and "no checks reported" in out.lower():
            return "none"
        try:
            states = {c.get("state", "").upper() for c in json.loads(out or "[]")}
        except json.JSONDecodeError:
            return "none" if code == 0 else "fail"
        if not states:
            return "none"
        if states & {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"}:
            return "fail"
        if states & {"PENDING", "QUEUED", "IN_PROGRESS", "EXPECTED", "WAITING"}:
            return "pending"
        return "pass"

    # Merge with a merge commit (no squash, no branch deletion: the runtime branch lives on).
    def merge(self, pr: int) -> bool:
        """Merge a PR; returns True on success."""
        return self._gh("pr", "merge", str(pr), "--merge")[0] == 0


# Run a git command in the worktree; returns (exit code, combined output).
def _git(wt: str, *args: str) -> tuple[int, str]:
    """Run git in a worktree and return its exit code and output."""
    r = subprocess.run(["git", "-C", wt, *args], capture_output=True, text=True, timeout=120)
    return r.returncode, (r.stdout + r.stderr).strip()


# Merge origin/base into the run branch so the PR diff contains only the bot's changes.
def _sync_with_base(wt: str, base: str) -> str | None:
    """Fetch and merge origin/<base> into the branch; returns an error string, or None."""
    if _git(wt, "fetch", "-q", "origin", base)[0] != 0:
        return "fetch failed"
    code, out = _git(wt, "merge", "-q", "--no-edit", f"origin/{base}")
    if code != 0:
        _git(wt, "merge", "--abort")
        return "merge with origin/main conflicted (left for human review): " + out[-200:]
    return None


# Level-2 integration: sync, push, PR, merge if the path policy and checks allow it.
def integrate(name: str, block: dict, worktree: str, summary: str, forge: Forge | None = None) -> dict:
    """Bring a completed run into origin/<base>, auto-merging only allowlisted, non-protected changes.

    Args:
        name: Automation name (used in the PR title).
        block: The definition's `integrate` block.
        worktree: The run's worktree path.
        summary: One-line run summary for the PR body.
        forge: GitHub operations (default: gh-backed Forge for the worktree).

    Returns:
        {stage, pr, merged, reason, files}: how far integration got and why it stopped there.
    """
    base = block.get("base", "main")
    forge = forge or Forge(worktree)
    res = {"stage": "sync", "pr": None, "merged": False, "reason": "", "files": []}
    err = _sync_with_base(worktree, base)
    if err:
        return {**res, "reason": err}
    branch = _git(worktree, "rev-parse", "--abbrev-ref", "HEAD")[1]
    files = [f for f in _git(worktree, "diff", "--name-only", f"origin/{base}...HEAD")[1].splitlines() if f]
    res.update(stage="push", files=files)
    if not files:
        return {**res, "reason": "nothing new to integrate"}
    code, out = _git(worktree, "push", "-q", "-u", "origin", branch)
    if code != 0:
        return {**res, "reason": "push failed: " + out[-200:]}
    body = (f"Opened by Sanctuary Bot automation `{name}`.\n\nLatest run: {summary}\n\nFiles: "
            + ", ".join(files[:30]) + ("\n\nAuto-merge allowed only for: " + ", ".join(block.get("auto_merge_paths", []))))
    pr = forge.find_or_create_pr(branch, base, f"{name}: automated update", body)
    res.update(stage="pr", pr=pr)
    if pr is None:
        return {**res, "reason": "could not open a pull request"}
    ok, why = merge_decision(files, block.get("auto_merge_paths", []))
    if not ok:
        return {**res, "reason": why}
    checks = forge.checks_state(pr)
    if checks in ("fail", "pending"):
        return {**res, "reason": f"PR checks {checks}: left open"}
    if not forge.merge(pr):
        return {**res, "reason": "merge failed: left open"}
    return {**res, "stage": "merged", "merged": True, "reason": f"auto-merged (checks: {checks})"}

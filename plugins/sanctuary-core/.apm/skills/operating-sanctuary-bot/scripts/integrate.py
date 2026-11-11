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
    Remote preservation is opt-in via the hash-approved preserve_remote flag. Its separate
    lifecycle path publishes only fixed generated exports and receipts, never arbitrary sources.

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
    - _git(), _sync_with_base(): Git execution and conflict-safe synchronization.
    - finish_delivery(): Verify post-merge workflows, local checkout and regeneration.
    - Forge.workflow_state(): Check workflows for an exact revision.
    - integrate(): Sync with base, push, open/update the PR, merge and finish delivery.
"""
from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
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
    if "preserve_remote" in block and not isinstance(block["preserve_remote"], bool):
        errs.append("integrate.preserve_remote must be a boolean")
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
        return r.returncode, (r.stdout or r.stderr).strip()

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
            return "none" if code == 0 else "fail"
        if states & {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"}:
            return "fail"
        if states & {"PENDING", "QUEUED", "IN_PROGRESS", "EXPECTED", "WAITING"}:
            return "pending"
        return "pass"

    # Merge with a merge commit (no squash, no branch deletion: the runtime branch lives on).
    def merge(self, pr: int) -> bool:
        """Merge a PR; returns True on success."""
        return self._gh("pr", "merge", str(pr), "--merge")[0] == 0


    # Inspect workflows for the integrated revision; errors are never treated as no checks.
    def workflow_state(self, revision: str, repository: str | None = None) -> str:
        """Return pass, none, pending or fail for workflows at an exact revision."""
        args = ["run", "list", "--commit", revision, "--limit", "100", "--json", "status,conclusion"]
        if repository:
            args += ["--repo", repository]
        code, out = self._gh(*args)
        if code:
            return "fail"
        try:
            runs = json.loads(out)
        except (ValueError, TypeError):
            return "fail"
        if not runs:
            endpoint = f"repos/{repository}/actions/workflows" if repository else "repos/{owner}/{repo}/actions/workflows"
            code, out = self._gh("api", endpoint, "--paginate")
            try:
                workflows = json.loads(out).get("workflows", [])
            except (ValueError, AttributeError):
                return "fail"
            if code:
                return "fail"
            return "pending" if any(w.get("state") == "active" for w in workflows) else "none"
        if len(runs) >= 100:
            return "pending"
        if any(r.get("status") != "completed" for r in runs):
            return "pending"
        return "pass" if all(r.get("conclusion") in ("success", "skipped", "neutral") for r in runs) else "fail"


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


# Finish delivery without switching or overwriting an occupied checkout.
def finish_delivery(worktree: str, base: str, forge: Forge, preserve_remote: bool = False) -> dict:
    """Verify workflows, fast-forward clean local base, and run supported regeneration.

    Returns:
        Separate CI, checkout and regeneration evidence; complete is false on any gap.
        Regenerated changes remain in the runtime branch for a reviewed follow-up PR.
    """
    result = {"complete": False, "stage": "post_merge", "reason": "", "ci": {}}
    if _git(worktree, "fetch", "-q", "origin", base)[0]:
        return {**result, "reason": "post-merge fetch failed"}
    revision = _git(worktree, "rev-parse", f"origin/{base}")[1]
    result["revision"] = revision
    code, common = _git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if code:
        return {**result, "reason": "cannot locate primary checkout"}
    primary = Path(common).parent
    result["checkout"] = str(primary)
    if preserve_remote:
        from preserve import wait_state
        result["ci"]["source"] = wait_state(lambda: forge.workflow_state(revision))
    else:
        result["ci"]["source"] = forge.workflow_state(revision)
    companion = primary.parent / "agent-plugins-skills"
    if companion.exists():
        code, out = _git(str(companion), "fetch", "-q", "origin", "main")
        if code:
            return {**result, "reason": "companion fetch failed: " + out[-200:]}
        companion_revision = _git(str(companion), "rev-parse", "origin/main")[1]
        probe = lambda: forge.workflow_state(companion_revision, "richfrem/agent-plugins-skills")
        result["ci"]["companion"] = wait_state(probe) if preserve_remote else probe()
    else:
        result["ci"]["companion"] = "unverified"
    if any(state not in ("pass", "none") for state in result["ci"].values()):
        return {**result, "reason": "post-merge workflows incomplete: " + str(result["ci"])}
    if _git(str(primary), "symbolic-ref", "--short", "HEAD") != (0, base):
        return {**result, "reason": "primary checkout is not on " + base + "; left untouched"}
    if _git(str(primary), "status", "--porcelain")[1]:
        return {**result, "reason": "primary checkout has uncommitted work; left untouched"}
    code, out = _git(str(primary), "merge", "--ff-only", f"origin/{base}")
    if code:
        return {**result, "reason": "primary fast-forward failed: " + out[-200:]}
    if _git(str(primary), "rev-parse", "HEAD")[1] != revision:
        return {**result, "reason": "primary revision verification failed"}
    result["local_synced"] = True
    if _git(worktree, "status", "--porcelain")[1]:
        return {**result, "reason": "runtime checkout dirty; regeneration left pending"}
    code, out = _git(worktree, "merge", "--ff-only", f"origin/{base}")
    if code:
        return {**result, "reason": "runtime fast-forward failed: " + out[-200:]}
    script = Path(worktree) / "plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/merge_prep.py"
    if not script.is_file():
        return {**result, "reason": "supported regeneration command missing"}
    proc = subprocess.run([sys.executable, str(script), "regenerate", "--root", worktree, "--format", "json"],
                          cwd=worktree, capture_output=True, text=True, timeout=300)
    result["regeneration"] = {"exit_code": proc.returncode, "output": (proc.stdout + proc.stderr)[-4000:]}
    if proc.returncode:
        return {**result, "reason": "post-merge regeneration failed; inspect regeneration evidence"}
    if preserve_remote:
        from preserve import preserve
        return {**result, **preserve(worktree, str(primary), base, forge)}
    changes = _git(worktree, "status", "--porcelain")[1]
    if changes:
        return {**result, "reason": "local checkout synchronized; regeneration produced changes requiring a scoped follow-up PR",
                "pending_changes": changes}
    return {**result, "complete": True, "stage": "delivered", "reason": "post-merge workflows verified; local checkout synchronized; regeneration complete"}


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
    res = {"complete": False, "stage": "sync", "pr": None, "merged": False, "reason": "", "files": []}
    if block.get("preserve_remote") is True:
        from preserve import LIFECYCLE, changed_paths, preserve
        code, common = _git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if code:
            return {**res, "reason": "cannot locate primary checkout for preservation"}
        dirty = changed_paths(worktree)
        ahead = set(_git(worktree, "diff", "--name-only", f"origin/{base}...HEAD")[1].splitlines())
        if (dirty or ahead) and (dirty | ahead) <= LIFECYCLE:
            if _git(worktree, "fetch", "-q", "origin", base)[0]:
                return {**res, "reason": "preservation resume fetch failed"}
            return {**res, **preserve(worktree, str(Path(common).parent), base, forge)}
    err = _sync_with_base(worktree, base)
    if err:
        return {**res, "reason": err}
    branch = _git(worktree, "rev-parse", "--abbrev-ref", "HEAD")[1]
    files = [f for f in _git(worktree, "diff", "--name-only", f"origin/{base}...HEAD")[1].splitlines() if f]
    res.update(stage="push", files=files)
    if not files:
        return {**res, **finish_delivery(worktree, base, forge, block.get("preserve_remote") is True)}
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
    return {**res, "merged": True, **finish_delivery(worktree, base, forge, block.get("preserve_remote") is True)}

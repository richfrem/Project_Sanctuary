#!/usr/bin/env python3
"""test_integrate.py - tests for integrate.py (level-2 integration of automation runs).

Purpose:
    Prove the auto-merge policy: only allowlisted, non-protected files merge; mission manifests,
    governance files and any code file always stay on an open PR. Exercises real git (a local
    bare repository as `origin`) with a fake GitHub layer, so no network or gh call is made.

Usage:
    python3 test_integrate.py

Key Input Dependencies:
    - integrate.py (sibling); git on PATH; temporary repositories

Key Functions:
    - class FakeForge: records PR/merge calls; configurable check state.
    - class Policy: deny rules, merge decision, integrate-block validation.
    - class EndToEnd: sync, push, PR and merge against a bare origin.
"""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import integrate as I  # noqa: E402

ALLOW = ["LEARNING/research/arxiv_daily/**"]


# Stand-in for the GitHub layer: no network, records what integrate asked for.
class FakeForge:
    """Records PR creation and merges; returns a configurable checks state."""

    # Configure the checks state the fake will report.
    def __init__(self, checks: str = "none", worktree: Path | None = None, workflow: str = "pass") -> None:
        """Start with no PRs and the given checks state."""
        self.checks, self.created, self.merged = checks, [], []
        self.worktree, self.workflow = worktree, workflow

    # Pretend to open PR #7 for the branch.
    def find_or_create_pr(self, branch: str, base: str, title: str, body: str) -> int:
        """Record the PR request and return a fixed number."""
        self.created.append((branch, base, title))
        return 7

    # Report the configured checks state.
    def checks_state(self, pr: int) -> str:
        """Return the configured state."""
        return self.checks

    # Record a merge.
    def merge(self, pr: int) -> bool:
        """Record the merge and report success."""
        self.merged.append(pr)
        if self.worktree:
            git(self.worktree, "push", "-q", "origin", "HEAD:main")
        return True

    # Return controlled post-merge workflow evidence.
    def workflow_state(self, revision: str, repository: str | None = None) -> str:
        """Return the fixture's post-merge workflow state."""
        return self.workflow


# Run git and fail loudly on error.
def git(cwd: Path, *args: str) -> str:
    """Run git in `cwd` and return stdout."""
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout


class Policy(unittest.TestCase):
    """Deny rules, merge decision and validation, without git."""

    def test_denied_rules(self) -> None:
        """Code files, missions, governance and plugins are always denied; data is not."""
        self.assertEqual(I.denied("LEARNING/research/arxiv_daily/pipeline/fetch_arxiv.py"), "code file")
        self.assertTrue(I.denied("LEARNING/missions/MISSION-001.json"))
        for p in (".agent/rules/constitution.md", "AGENTS.md", "protocols/128.md", "plugins/x/SKILL.md",
                  "LEARNING/index.jsonl", "LEARNING/inbox.jsonl", "huggingface/data/soul_traces.jsonl", "run.sh"):
            self.assertTrue(I.denied(p), p)
        self.assertIsNone(I.denied("LEARNING/research/arxiv_daily/analysis/2026-11-09.md"))

    def test_merge_decision(self) -> None:
        """Only allowlisted data merges; code inside the allowlist and anything outside go to review."""
        data = ["LEARNING/research/arxiv_daily/latest.md", "LEARNING/research/arxiv_daily/seeds.jsonl"]
        self.assertEqual(I.merge_decision(data, ALLOW), (True, "ok"))
        ok, why = I.merge_decision(data + ["LEARNING/research/arxiv_daily/pipeline/make_digest.py"], ALLOW)
        self.assertFalse(ok)
        self.assertIn("code file", why)
        ok, why = I.merge_decision(data + ["LEARNING/missions/MISSION-068.json"], ALLOW)
        self.assertFalse(ok)
        self.assertIn("protected", why)
        ok, why = I.merge_decision(["LEARNING/research/other/notes.md"], ALLOW)
        self.assertFalse(ok)
        self.assertIn("outside the allowlist", why)
        self.assertFalse(I.merge_decision([], ALLOW)[0])
        self.assertFalse(I.merge_decision(data, [])[0])

    def test_validate_integrate(self) -> None:
        """Allowlists must name a dossier under LEARNING/research/."""
        self.assertEqual(I.validate_integrate({"auto_merge_paths": ALLOW}), [])
        self.assertEqual(I.validate_integrate(None), [])
        for bad in (["**"], ["plugins/**"], ["LEARNING/research/**"], ["LEARNING/research/../missions/**"], ["LEARNING/missions/**"]):
            self.assertTrue(I.validate_integrate({"auto_merge_paths": bad}), bad)


class WorkflowEvidence(unittest.TestCase):
    """Empty workflow results do not conceal a dispatch delay or API error."""

    def test_active_workflow_without_run_is_pending(self) -> None:
        """Wait for active workflows rather than claiming none before dispatch."""
        forge = I.Forge("unused")
        with patch.object(forge, "_gh", side_effect=[(0, "[]"), (0, '{"workflows":[{"state":"active"}]}')]):
            self.assertEqual(forge.workflow_state("abc"), "pending")

    def test_no_configured_workflows_is_explicit_none(self) -> None:
        """Only a successful empty workflow inventory establishes no workflows."""
        forge = I.Forge("unused")
        with patch.object(forge, "_gh", side_effect=[(0, "[]"), (0, '{"workflows":[]}')]):
            self.assertEqual(forge.workflow_state("abc"), "none")
        with patch.object(forge, "_gh", return_value=(1, "")):
            self.assertEqual(forge.workflow_state("abc"), "fail")
            self.assertEqual(forge.checks_state(7), "fail")


class EndToEnd(unittest.TestCase):
    """integrate() against a real bare `origin` and a fake GitHub layer."""

    def setUp(self) -> None:
        """Create origin (bare), a main clone, and a runtime worktree on its own branch."""
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.origin = base / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)], check=True)
        self.main = base / "main"
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.main)], check=True, capture_output=True)
        for c in (["config", "user.email", "t@t"], ["config", "user.name", "t"]):
            git(self.main, *c)
        (self.main / "README.md").write_text("x\n")
        regen = self.main / "plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/merge_prep.py"
        regen.parent.mkdir(parents=True)
        regen.write_text("print('{}')\n")
        git(self.main, "add", "-A")
        git(self.main, "commit", "-qm", "init")
        git(self.main, "push", "-q", "origin", "main")
        subprocess.run(["git", "clone", "-q", str(self.origin), str(base / "agent-plugins-skills")],
                       check=True, capture_output=True)
        self.wt = base / "wt"
        git(self.main, "worktree", "add", "-q", "-b", "bot/x", str(self.wt))

    def tearDown(self) -> None:
        """Remove the temporary repositories."""
        self.tmp.cleanup()

    # Commit a file in the runtime worktree.
    def commit(self, rel: str, text: str = "data\n") -> None:
        """Write and commit one file on the runtime branch."""
        p = self.wt / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        git(self.wt, "add", "-A")
        git(self.wt, "commit", "-qm", f"add {rel}")

    def test_allowlisted_data_auto_merges(self) -> None:
        """Data inside the allowlist is pushed, a PR opened, and merged."""
        self.commit("LEARNING/research/arxiv_daily/analysis/d.md")
        f = FakeForge(worktree=self.wt)
        r = I.integrate("arxiv-daily", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", f)
        self.assertTrue(r["merged"], r)
        self.assertEqual(f.merged, [7])
        self.assertTrue(r["complete"], r)
        self.assertTrue((self.main / "LEARNING/research/arxiv_daily/analysis/d.md").is_file())
        self.assertEqual(git(self.main, "rev-parse", "HEAD"), git(self.origin, "rev-parse", "main"))
        self.assertIn("bot/x", git(self.origin, "branch"))

    def test_code_in_allowlist_stays_open(self) -> None:
        """A code file inside the allowed folder opens a PR but is never merged."""
        self.commit("LEARNING/research/arxiv_daily/pipeline/new.py", "print(1)\n")
        f = FakeForge(worktree=self.wt)
        r = I.integrate("arxiv-daily", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", f)
        self.assertFalse(r["merged"])
        self.assertEqual((r["pr"], f.merged), (7, []))
        self.assertIn("code file", r["reason"])

    def test_mission_manifest_stays_open(self) -> None:
        """A mission manifest change is never auto-merged."""
        self.commit("LEARNING/research/arxiv_daily/latest.md")
        self.commit("LEARNING/missions/MISSION-999.json", "{}\n")
        f = FakeForge(worktree=self.wt)
        r = I.integrate("arxiv-daily", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", f)
        self.assertFalse(r["merged"])
        self.assertEqual(f.merged, [])

    def test_pending_checks_block_merge(self) -> None:
        """Pending or failing PR checks leave the PR open."""
        self.commit("LEARNING/research/arxiv_daily/latest.md")
        for state in ("pending", "fail"):
            f = FakeForge(checks=state)
            r = I.integrate("arxiv-daily", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", f)
            self.assertFalse(r["merged"], state)
            self.assertEqual(f.merged, [])

    def test_conflict_with_main_is_left_for_review(self) -> None:
        """A merge conflict with origin/main aborts cleanly and is reported."""
        self.commit("LEARNING/research/arxiv_daily/latest.md", "bot\n")
        (self.main / "LEARNING/research/arxiv_daily").mkdir(parents=True)
        (self.main / "LEARNING/research/arxiv_daily/latest.md").write_text("human\n")
        git(self.main, "add", "-A")
        git(self.main, "commit", "-qm", "human edit")
        git(self.main, "push", "-q", "origin", "main")
        r = I.integrate("arxiv-daily", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", FakeForge())
        self.assertFalse(r["merged"])
        self.assertIn("conflicted", r["reason"])
        self.assertEqual(git(self.wt, "status", "--porcelain"), "")

    def test_nothing_new(self) -> None:
        """No commits beyond origin/main: nothing is pushed or opened."""
        f = FakeForge(worktree=self.wt)
        r = I.integrate("arxiv-daily", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", f)
        self.assertEqual((r["merged"], f.created), (False, []))

    def test_dirty_primary_preserved(self) -> None:
        """Merged delivery stays partial and preserves unrelated local edits."""
        self.commit("LEARNING/research/arxiv_daily/latest.md")
        (self.main / "README.md").write_text("human edit\n")
        r = I.integrate("x", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", FakeForge(worktree=self.wt))
        self.assertTrue(r["merged"])
        self.assertFalse(r["complete"])
        self.assertIn("uncommitted", r["reason"])
        self.assertEqual((self.main / "README.md").read_text(), "human edit\n")

    def test_primary_other_branch_preserved(self) -> None:
        """An occupied primary checkout is never switched to main."""
        git(self.main, "checkout", "-qb", "human")
        self.commit("LEARNING/research/arxiv_daily/latest.md")
        r = I.integrate("x", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", FakeForge(worktree=self.wt))
        self.assertFalse(r["complete"])
        self.assertEqual(git(self.main, "branch", "--show-current").strip(), "human")

    def test_post_merge_ci_failure_is_partial(self) -> None:
        """A successful merge with failing post-merge CI cannot claim delivery."""
        self.commit("LEARNING/research/arxiv_daily/latest.md")
        r = I.integrate("x", {"auto_merge_paths": ALLOW}, str(self.wt), "ok",
                        FakeForge(worktree=self.wt, workflow="fail"))
        self.assertTrue(r["merged"])
        self.assertFalse(r["complete"])
        self.assertEqual(r["ci"]["source"], "fail")

    def test_no_new_data_recovers_missed_local_sync(self) -> None:
        """No-change runs still deliver an already merged digest locally."""
        self.commit("LEARNING/research/arxiv_daily/latest.md")
        git(self.wt, "push", "-q", "origin", "HEAD:main")
        r = I.integrate("x", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", FakeForge(worktree=self.wt))
        self.assertTrue(r["complete"], r)
        self.assertTrue((self.main / "LEARNING/research/arxiv_daily/latest.md").exists())

    def test_regeneration_failure_is_partial(self) -> None:
        """A failed supported regeneration command cannot count as complete."""
        script = self.wt / "plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/merge_prep.py"
        script.write_text("raise SystemExit(2)\n")
        git(self.wt, "add", "-A")
        git(self.wt, "commit", "-qm", "fixture failure")
        git(self.wt, "push", "-q", "origin", "HEAD:main")
        r = I.integrate("x", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", FakeForge(worktree=self.wt))
        self.assertFalse(r["complete"])
        self.assertIn("regeneration failed", r["reason"])

    def test_regenerated_changes_remain_pending(self) -> None:
        """Generated exports stay preserved on the runtime branch for review."""
        script = self.wt / "plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/merge_prep.py"
        script.write_text("from pathlib import Path\nPath('generated.txt').write_text('export')\nprint('{}')\n")
        git(self.wt, "add", "-A")
        git(self.wt, "commit", "-qm", "fixture generation")
        git(self.wt, "push", "-q", "origin", "HEAD:main")
        r = I.integrate("x", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", FakeForge(worktree=self.wt))
        self.assertFalse(r["complete"])
        self.assertIn("generated.txt", r["pending_changes"])
        self.assertFalse((self.main / "generated.txt").exists())
        self.assertEqual((self.wt / "generated.txt").read_text(), "export")

    def test_diverged_primary_preserves_commits(self) -> None:
        """Unpublished main commits are never reset or merged automatically."""
        (self.main / "human.txt").write_text("retain")
        git(self.main, "add", "human.txt")
        git(self.main, "commit", "-qm", "unpublished human change")
        original = git(self.main, "rev-parse", "HEAD")
        self.commit("LEARNING/research/arxiv_daily/latest.md")
        r = I.integrate("x", {"auto_merge_paths": ALLOW}, str(self.wt), "ok", FakeForge(worktree=self.wt))
        self.assertFalse(r["complete"])
        self.assertIn("fast-forward failed", r["reason"])
        self.assertEqual(git(self.main, "rev-parse", "HEAD"), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)

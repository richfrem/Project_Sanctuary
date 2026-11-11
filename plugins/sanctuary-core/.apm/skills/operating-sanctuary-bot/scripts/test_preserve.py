#!/usr/bin/env python3
"""Exercise bot preservation with real git and fake remote commands (no network).

Purpose: prove ordering, exact-path publication, remote verification and resumability.
Usage: python3 test_preserve.py
Key Input Dependencies: preserve.py, test_integrate fixtures, git, unittest.
Key Functions: Preservation.setUp(), tearDown(), run_command(), receipt(),
    test_full_cycle(), test_unexpected_source_refused(), test_failed_upload(),
    test_invalid_export(), test_pending_pr(), test_committed_export_resume(),
    test_unverified_receipt(), test_already_persisted_no_upload(),
    test_main_never_committed(), test_primary_edits_preserved().
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integrate as I
import preserve as P
import test_integrate as fixtures


# Reuse real temporary repository setup without inheriting its tests.
class Preservation(unittest.TestCase):
    """Exercise the complete publication/sync/receipt cycle with independent git checks."""

    def setUp(self) -> None:
        """Create a published baseline export and injected remote-command recorder."""
        fixtures.EndToEnd.setUp(self)
        for rel in ("huggingface/data/soul_traces.jsonl", "huggingface/metadata/coverage_manifest.json",
                    "huggingface/metadata/manifest.json"):
            path = self.wt / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n')
        fixtures.git(self.wt, "add", "-A")
        fixtures.git(self.wt, "commit", "-qm", "baseline export")
        fixtures.git(self.wt, "push", "-q", "origin", "HEAD:main")
        fixtures.git(self.main, "pull", "--ff-only")
        self.forge = fixtures.FakeForge(worktree=self.wt)
        self.calls, self.synced, self.upload_failure, self.invalid, self.bad_receipt = [], False, False, False, False
        self.init = patch.object(P, "init_command", return_value=["hf-init"])
        self.init.start()

    def tearDown(self) -> None:
        """Release mocks and temporary repositories."""
        self.init.stop()
        fixtures.EndToEnd.tearDown(self)

    def receipt(self) -> None:
        """Write a receipt binding the two actual fixture files, as the sync tool does."""
        record = {"verified": not self.bad_receipt, "coverage_required_ok": True, "revision": "hub-revision"}
        for key, rel in (("dataset", "huggingface/data/soul_traces.jsonl"),
                         ("manifest", "huggingface/metadata/coverage_manifest.json")):
            digest = hashlib.sha256((self.wt / rel).read_bytes()).hexdigest()
            record[f"local_{key}_sha256"] = record[f"remote_{key}_sha256"] = digest
        path = self.wt / P.RECEIPT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record) + '\n')

    def run_command(self, argv: list[str], cwd: str, timeout: int = 300) -> tuple[int, str]:
        """Model supported command contracts while checking export publication precedes sync."""
        self.calls.append(argv)
        if any("soul_coverage.py" in arg for arg in argv):
            return (3 if self.invalid else 0 if self.synced else 4), "coverage evidence"
        if any("sync_soul_dataset.py" in arg for arg in argv):
            # The public base must contain the exact export before any upload is attempted.
            public = fixtures.git(self.origin, "show", "main:huggingface/data/soul_traces.jsonl")
            self.assertEqual(public, (self.wt / "huggingface/data/soul_traces.jsonl").read_text())
            if self.upload_failure:
                return 1, "upload failed"
            self.synced = True
            self.receipt()
        return 0, "ok"

    def test_full_cycle(self) -> None:
        """Publish export first, then upload, verify, publish receipt and synchronize main."""
        (self.wt / "huggingface/data/soul_traces.jsonl").write_text('{"new":true}\n')
        result = P.preserve(str(self.wt), str(self.main), "main", self.forge, self.run_command)
        self.assertTrue(result["complete"], result)
        self.assertEqual(len(result["preservation_prs"]), 2)
        self.assertEqual((self.main / P.RECEIPT).read_bytes(), (self.wt / P.RECEIPT).read_bytes())
        self.assertEqual(fixtures.git(self.main, "status", "--porcelain"), "")
        self.assertEqual(fixtures.git(self.wt, "status", "--porcelain"), "")
        self.assertEqual(result["remote_revision"], "hub-revision")

    def test_unexpected_source_refused(self) -> None:
        """A source edit cannot enter the generated-file exception."""
        (self.wt / "README.md").write_text('unexpected source\n')
        result = P.preserve(str(self.wt), str(self.main), "main", self.forge, self.run_command)
        self.assertFalse(result["complete"])
        self.assertIn("unexpected source", result["reason"])
        self.assertEqual(self.forge.created, [])
        self.assertFalse(self.synced)

    def test_failed_upload(self) -> None:
        """An upload failure cannot become preserved or publish a success receipt."""
        self.upload_failure = True
        result = P.preserve(str(self.wt), str(self.main), "main", self.forge, self.run_command)
        self.assertFalse(result["complete"])
        self.assertIn("synchronization", result["reason"])
        self.assertFalse((self.main / P.RECEIPT).exists())

    def test_invalid_export(self) -> None:
        """Invalid schema/stale coverage is refused before publication and upload."""
        self.invalid = True
        result = P.preserve(str(self.wt), str(self.main), "main", self.forge, self.run_command)
        self.assertFalse(result["complete"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.forge.created, [])

    def test_pending_pr(self) -> None:
        """Pending PR checks retain the committed export without attempting upload."""
        (self.wt / "huggingface/data/soul_traces.jsonl").write_text('{"pending":true}\n')
        self.forge.checks = "pending"
        with patch.object(P, "wait_state", side_effect=lambda check: check()):
            result = P.preserve(str(self.wt), str(self.main), "main", self.forge, self.run_command)
        self.assertFalse(result["complete"])
        self.assertEqual(fixtures.git(self.wt, "status", "--porcelain"), "")
        self.assertFalse(self.synced)

    def test_committed_export_resume(self) -> None:
        """Integration resumes a lifecycle-only branch instead of rejecting its protected paths."""
        (self.wt / "huggingface/data/soul_traces.jsonl").write_text('{"resume":true}\n')
        fixtures.git(self.wt, "add", "-A")
        fixtures.git(self.wt, "commit", "-qm", "pending generated export")
        original = P.preserve
        with patch.object(P, "preserve", side_effect=lambda *args: original(*args, run=self.run_command)):
            result = I.integrate("x", {"auto_merge_paths": fixtures.ALLOW, "preserve_remote": True},
                                 str(self.wt), "resume", self.forge)
        self.assertTrue(result["complete"], result)
        self.assertTrue((self.main / P.RECEIPT).exists())

    def test_unverified_receipt(self) -> None:
        """A tool exit code alone cannot substitute for a verified two-hash receipt."""
        self.bad_receipt = True
        result = P.preserve(str(self.wt), str(self.main), "main", self.forge, self.run_command)
        self.assertFalse(result["complete"])
        self.assertIn("both current hashes", result["reason"])

    def test_already_persisted_no_upload(self) -> None:
        """An unchanged verified export does not trigger another upload or new commit."""
        self.synced = True
        self.receipt()
        fixtures.git(self.wt, "add", "-A")
        fixtures.git(self.wt, "commit", "-qm", "prior verified receipt")
        fixtures.git(self.wt, "push", "-q", "origin", "HEAD:main")
        fixtures.git(self.main, "pull", "--ff-only")
        original = fixtures.git(self.wt, "rev-parse", "HEAD")
        result = P.preserve(str(self.wt), str(self.main), "main", self.forge, self.run_command)
        self.assertTrue(result["complete"], result)
        self.assertEqual(fixtures.git(self.wt, "rev-parse", "HEAD"), original)
        self.assertFalse(any("sync_soul_dataset.py" in arg for call in self.calls for arg in call))

    def test_main_never_committed(self) -> None:
        """Passing main as the runtime checkout never creates a main commit."""
        before = fixtures.git(self.main, "rev-parse", "HEAD")
        result = P.preserve(str(self.main), str(self.main), "main", self.forge, self.run_command)
        self.assertFalse(result["complete"])
        self.assertEqual(fixtures.git(self.main, "rev-parse", "HEAD"), before)
        self.assertEqual(self.calls, [])

    def test_primary_edits_preserved(self) -> None:
        """Occupied main blocks remote preservation without touching user edits."""
        (self.main / "README.md").write_text("human change")
        result = P.preserve(str(self.wt), str(self.main), "main", self.forge, self.run_command)
        self.assertFalse(result["complete"])
        self.assertEqual((self.main / "README.md").read_text(), "human change")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

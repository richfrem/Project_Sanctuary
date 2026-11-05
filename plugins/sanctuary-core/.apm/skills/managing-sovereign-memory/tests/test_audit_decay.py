"""audit_decay must age undated dossiers by their last git commit, not by file mtime (a clone resets mtime, so decay never fired)."""
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_decay  # noqa: E402


def _git(cwd, *a, when=None):
    env = dict(os.environ)
    if when:
        env.update(GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    subprocess.run(["git", *a], cwd=str(cwd), check=True, capture_output=True, env=env)


def _repo(days_ago, body):
    root = Path(tempfile.mkdtemp()).resolve()
    _git(root, "init", "-q", "-b", "main"); _git(root, "config", "user.email", "t@t"); _git(root, "config", "user.name", "t")
    d = root / "LEARNING" / "concepts"; d.mkdir(parents=True)
    f = d / "c.md"; f.write_text(body)
    _git(root, "add", "-A")
    when = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S+0000")
    _git(root, "commit", "-qm", "c", when=when)
    return root, f


class TestAuditDecay(unittest.TestCase):
    def test_undated_confirmed_dossier_decays_by_its_last_commit_date_even_though_mtime_is_now(self):
        root, f = _repo(90, "# C\n**Status:** CONFIRMED\n")
        self.assertEqual(audit_decay.audit_concepts(root, apply_fixes=False), 1)

    def test_recently_committed_undated_dossier_does_not_decay(self):
        root, f = _repo(2, "# C\n**Status:** CONFIRMED\n")
        self.assertEqual(audit_decay.audit_concepts(root, apply_fixes=False), 0)

    def test_explicit_last_verified_wins_over_the_commit_date(self):
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        root, f = _repo(90, f"# C\n**Status:** CONFIRMED\n**Last Verified:** {yesterday}\n")
        self.assertEqual(audit_decay.audit_concepts(root, apply_fixes=False), 0)

    def test_outside_git_falls_back_to_mtime(self):
        root = Path(tempfile.mkdtemp()).resolve()
        d = root / "LEARNING" / "concepts"; d.mkdir(parents=True)
        (d / "c.md").write_text("# C\n**Status:** CONFIRMED\n")
        self.assertEqual(audit_decay.audit_concepts(root, apply_fixes=False), 0)  # fresh file, no git

    def test_apply_downgrades_a_decayed_undated_dossier(self):
        root, f = _repo(90, "# C\n**Status:** CONFIRMED\n")
        audit_decay.audit_concepts(root, apply_fixes=True)
        self.assertIn("**Status:** OBSERVED", f.read_text())


if __name__ == "__main__":
    unittest.main()

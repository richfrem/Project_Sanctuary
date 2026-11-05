"""Tests for the rebuildable chronicle ledger (MISSION-034 follow-up)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import chronicle_manager as cm  # noqa: E402


def _entry(d, name, title, date="2026-11-04", status="draft"):
    (d / name).write_text(f"# Living Chronicle - Entry\n\n**Title:** {title}\n**Date:** {date}\n**Status:** {status}\n\n---\n\nbody\n")


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.entries = self.root / "entries"
        self.entries.mkdir()

    def test_counts_every_numbered_entry_newest_first(self):
        _entry(self.entries, "001_a.md", "A"); _entry(self.entries, "010_b.md", "B"); _entry(self.entries, "362_c.md", "C")
        (self.entries / "README.md").write_text("not an entry"); (self.entries / ".hidden").write_text("x")
        led = cm.build_ledger(self.entries)
        self.assertEqual(led["total_entries"], 3)
        self.assertEqual([e["id"] for e in led["entries"]], [362, 10, 1])
        self.assertEqual(led["entries"][0]["title"], "C")

    def test_is_deterministic_and_writes_next_to_entries(self):
        _entry(self.entries, "005_x.md", "X")
        p = cm.write_ledger(self.entries)
        self.assertEqual(p, self.root / "ledger.json")
        first = p.read_bytes()
        cm.write_ledger(self.entries)
        self.assertEqual(p.read_bytes(), first)

    def test_new_entry_appears_after_create(self):
        _entry(self.entries, "001_a.md", "A")
        orig = cm.CHRONICLE_DIR
        cm.CHRONICLE_DIR = self.entries
        try:
            cm.create_entry("Second", "body", author="t")
        finally:
            cm.CHRONICLE_DIR = orig
        led = json.loads((self.root / "ledger.json").read_text())
        self.assertEqual(led["total_entries"], 2)
        self.assertEqual(led["entries"][0]["id"], 2)

    def test_repo_ledger_matches_entry_files(self):
        repo = Path(__file__).resolve().parents[6]
        stored = json.loads((repo / "chronicle" / "ledger.json").read_text())
        self.assertEqual(stored, cm.build_ledger(repo / "chronicle" / "entries"))


if __name__ == "__main__":
    unittest.main()

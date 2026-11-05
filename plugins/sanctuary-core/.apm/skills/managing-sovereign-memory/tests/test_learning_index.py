"""LEARNING/INDEX.md must describe the folder as it is, and no skill, script or core doc may point at a LEARNING/ path that does not exist.

Standing rule for any move under LEARNING/: update every reference under plugins/ and the core docs in the same PR; this test fails if one is missed.
"""
import os
import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
LEARNING = REPO / "LEARNING"
SCAN_ROOTS = ["plugins", "AGENTS.md", "START_HERE.md", "README.md", "architecture.md", "INSTALL.md", "context", ".agent/rules",
               ".agent/hooks", ".agent/learning/guardian_manifest.json", ".agent/learning/bootstrap_manifest.json", "protocols"]
SCAN_SUFFIXES = (".md", ".py", ".json", ".sh", ".mmd", ".yml", ".yaml", ".txt")
# tests and evals build their own fixture trees; their LEARNING/ paths are not references to this repository
SKIP_PARTS = ("/tests/", "relay_eval.py", "impact_eval.py", "agent_eval_tasks.py", "/evals/")
PATH_RX = re.compile(r"LEARNING/[A-Za-z0-9_\-./]+")


def _tracked(paths):
    out = subprocess.run(["git", "ls-files", *paths], cwd=str(REPO), capture_output=True, text=True).stdout.split("\n")
    return [f for f in out if f.endswith(SCAN_SUFFIXES) and not any(s in "/" + f for s in SKIP_PARTS)]


class TestLearningIndex(unittest.TestCase):
    def setUp(self):
        self.index = (LEARNING / "INDEX.md").read_text(encoding="utf-8")

    def test_every_top_level_entry_is_described_in_the_index(self):
        missing = [p.name for p in sorted(LEARNING.iterdir())
                   if not p.name.startswith(".") and p.name not in ("INDEX.md", "README.md") and p.name not in self.index]
        self.assertEqual(missing, [], f"INDEX.md does not mention: {missing}")

    def test_every_concept_dossier_is_listed(self):
        missing = [p.name for p in sorted((LEARNING / "concepts").glob("*.md")) if p.name not in self.index]
        self.assertEqual(missing, [], f"concepts missing from INDEX.md: {missing}")

    def test_index_links_resolve(self):
        broken = []
        for target in re.findall(r"\]\((\./[^)#]+)\)", self.index):
            if not (LEARNING / target).exists():
                broken.append(target)
        self.assertEqual(broken, [], broken)

    def test_no_reference_to_a_missing_learning_path_in_skills_scripts_or_core_docs(self):
        dangling = {}
        for f in _tracked(SCAN_ROOTS):
            try:
                text = (REPO / f).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for m in PATH_RX.findall(text):
                p = m.rstrip(".,;:)").rstrip("/")
                if p.endswith("-") or any(x in p for x in ("NNN", "XXX", "xxx", "<", "*", "{")):
                    continue  # partial or placeholder
                if not (REPO / p).exists():
                    dangling.setdefault(p, set()).add(f)
        self.assertEqual(dangling, {}, "\n".join(f"{p}  <- {sorted(fs)}" for p, fs in sorted(dangling.items())))


if __name__ == "__main__":
    unittest.main()

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
# retired paths that a file names on purpose (the stale-location guard reports them); everything else must exist
RETIRED_ON_PURPOSE = {
    "plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/manage_backlog.py": ("LEARNING/research/missions",),
}


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
                if p in RETIRED_ON_PURPOSE.get(f, ()):
                    continue
                if not (REPO / p).exists():
                    dangling.setdefault(p, set()).add(f)
        self.assertEqual(dangling, {}, "\n".join(f"{p}  <- {sorted(fs)}" for p, fs in sorted(dangling.items())))


if __name__ == "__main__":
    unittest.main()


ARCHIVE_ROOT = REPO / "ARCHIVE" / "LEARNING_pre_2026-11"
MOVED = ["security", "associative_architecture_brief.md", "calibration_log.json",
         "topics/scholarly_discourse", "topics/openclaw_moltbook_ecosystem", "topics/safe_agent_zero", "topics/documentation_link_remediation",
         "topics/protocol_128_remediation", "topics/liquid_neural_networks", "topics/forge_v5_evolution", "topics/knowledge_preservation_red_team"]


class TestTopicsAndArchive(unittest.TestCase):
    def test_every_research_dossier_has_a_status_row_in_the_research_readme(self):
        readme = (LEARNING / "research" / "README.md").read_text(encoding="utf-8")
        topics = [p.name for p in sorted((LEARNING / "research").iterdir()) if p.is_dir()]
        missing = [t for t in topics if f"`{t}`" not in readme]
        self.assertEqual(missing, [], f"research/README.md has no row for: {missing}")
        for line in readme.splitlines():
            if line.startswith("| `"):
                self.assertTrue(any(s in line for s in ("| KEEP |", "| HISTORICAL |")), line)

    def test_moved_material_is_in_the_archive_with_a_map_and_not_in_learning(self):
        readme = (ARCHIVE_ROOT / "README.md").read_text(encoding="utf-8")
        for item in MOVED:
            live = LEARNING / item.replace("topics/", "research/", 1)  # the dossiers folder was renamed topics -> research; the archive keeps the old layout
            self.assertTrue((ARCHIVE_ROOT / item).exists(), f"archive is missing {item}")
            self.assertFalse(live.exists(), f"{item} is still under LEARNING/")
            self.assertIn(item, readme, f"archive README does not map {item}")

    def test_archived_files_are_tracked_by_git(self):
        tracked = subprocess.run(["git", "ls-files", "ARCHIVE/LEARNING_pre_2026-11"], cwd=str(REPO), capture_output=True, text=True).stdout.split()
        self.assertGreater(len(tracked), 40)

    def test_no_kept_document_links_into_material_that_moved_to_the_archive(self):
        """Pasted external links in historical dossiers are already broken (not our concern); links INTO moved paths are the regression to catch."""
        moved_abs = [(LEARNING / m.replace("topics/", "research/", 1)).resolve() for m in MOVED]
        offenders = []
        for md in LEARNING.rglob("*.md"):
            if "/research/" in str(md):
                continue
            for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)\s#]+)", md.read_text(encoding="utf-8", errors="ignore")):
                if target.startswith(("/", "file:")):
                    continue
                dest = (md.parent / target).resolve()
                if any(dest == m or m in dest.parents for m in moved_abs):
                    offenders.append(f"{md.relative_to(REPO)} -> {target}")
        self.assertEqual(offenders, [], "\n".join(offenders[:15]))


class TestProceduresFolder(unittest.TestCase):
    """procedures/ is the home of the procedural-memory mechanism (record_procedure_outcome.py, three open missions); the three seed procedures were retired."""

    def test_the_retired_seed_procedures_are_gone_and_the_format_readme_stays(self):
        proc = LEARNING / "procedures"
        self.assertTrue((proc / "README.md").exists())
        self.assertEqual([p.name for p in proc.iterdir() if p.name != "README.md"], [])
        for retired in ("procedures/orchestration", "procedures/architecture"):
            self.assertTrue((ARCHIVE_ROOT / retired).exists(), retired)
            self.assertIn(retired, (ARCHIVE_ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertFalse((ARCHIVE_ROOT / "procedures" / "git").exists())  # deleted, not archived

    def test_the_outcome_recorder_does_not_cite_a_deleted_procedure_in_its_examples(self):
        text = (REPO / "plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/record_procedure_outcome.py").read_text(encoding="utf-8")
        self.assertNotIn("create_feature_branch", text)
        self.assertNotIn("proc-git-feature-branch", text)

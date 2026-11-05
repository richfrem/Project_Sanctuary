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


NAV_DOCS = ["AGENTS.md", "START_HERE.md", "README.md", "architecture.md", "INSTALL.md", "context", ".agent/rules", ".agent/learning/README.md"]
REPO_ROOTS = ("LEARNING", "plugins", "context", "protocols", "chronicle", "huggingface", "ARCHIVE", "ADRs", "docs", "specs", "scripts", "tools",
              "mcp_servers", "forge", "01_PROTOCOLS", "wiki", "handoffs", ".agent")
REPO_PATH_RX = re.compile(r"(?<![A-Za-z0-9_./\-~$])((?:%s)/[A-Za-z0-9_.\-/]*[A-Za-z0-9_\-/])" % "|".join(re.escape(r) for r in REPO_ROOTS))


class TestNavigationDocsAreCurrent(unittest.TestCase):
    """The files every agent reads first must not point at paths that do not exist or describe a layout that was replaced."""

    def _docs(self):
        out = subprocess.run(["git", "ls-files", *NAV_DOCS], cwd=str(REPO), capture_output=True, text=True).stdout.split()
        return [f for f in out if f.endswith((".md", ".json"))]

    def test_no_dangling_repo_paths_in_the_navigation_docs(self):
        dangling = {}
        for f in self._docs():
            text = (REPO / f).read_text(encoding="utf-8", errors="ignore")
            for m in REPO_PATH_RX.findall(text):
                p = m.rstrip("/.,;:)")
                if any(x in p for x in ("NNN", "<", "*", "{", "XXX", "xxx", "YYYY")) or p.startswith((".agent/learning/precompact", ".agent/coordination")):
                    continue  # placeholders and gitignored runtime state, which a fresh clone does not have
                if not (REPO / p).exists():
                    dangling.setdefault(p, set()).add(f)
        self.assertEqual(dangling, {}, "\n".join(f"{p}  <- {sorted(fs)}" for p, fs in sorted(dangling.items())))

    def test_docs_do_not_describe_the_retired_topics_folder_or_a_wrong_skill_count(self):
        actual = len([p for p in (REPO / "plugins/sanctuary-core/.apm/skills").iterdir() if p.is_dir()])
        problems = []
        for f in ("AGENTS.md", "architecture.md", "START_HERE.md", "context/soul.md", "context/memory.md"):
            text = (REPO / f).read_text(encoding="utf-8")
            for line in text.splitlines():
                if re.search(r"`topics/`|concepts/topics|/topics/", line) and not re.search(r"formerly|became|renamed|retired|was consolidated", line):
                    problems.append(f"{f}: describes the retired topics/ folder as current: {line.strip()[:90]}")
            for n in re.findall(r"\\b(\\d+) (?:core |APM-compliant |sovereign )?skills\\b", text):
                if int(n) != actual and f.startswith("context/"):
                    problems.append(f"{f}: says {n} skills, there are {actual}")
        self.assertEqual(problems, [], problems)

    def test_docs_say_to_reinstall_after_changing_the_plugin(self):
        for f in ("AGENTS.md", "INSTALL.md"):
            text = (REPO / f).read_text(encoding="utf-8")
            self.assertIn("apm install", text, f)
        self.assertRegex((REPO / "AGENTS.md").read_text(encoding="utf-8"), r"(?i)after (editing|changing).{0,80}plugins/sanctuary-core.{0,160}apm install")


class TestRetiredTreeReferences(unittest.TestCase):
    """01_PROTOCOLS became protocols/; forge/ was archived (ARCHIVE/forge) and its tooling moved into the forging-sovereign-models skill."""

    HISTORY = ("ARCHIVE/", "chronicle/", "huggingface/", "LEARNING/experiments/")
    SELF = ("plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/tests/test_learning_index.py",)
    # output locations the forge scripts create on first run (a gitignored workspace), not references to files that should exist
    FORGE_OUTPUTS = ("forge/data/train.jsonl", "forge/data/sanctuary_genome_train.jsonl", "forge/Modelfile")
    RX = re.compile(r"(?<![A-Za-z0-9_./\-~$])((?:01_PROTOCOLS|forge)/[A-Za-z0-9_.\-/]*[A-Za-z0-9_\-/])")

    def _hits(self):
        files = subprocess.run(["git", "ls-files"], cwd=str(REPO), capture_output=True, text=True).stdout.split("\n")
        for f in files:
            if not f or f.startswith(self.HISTORY) or f in self.SELF or not f.endswith((".md", ".py", ".json", ".sh", ".yml", ".yaml", ".txt", ".mmd")):
                continue
            try:
                text = (REPO / f).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for m in self.RX.findall(text):
                yield f, m.rstrip("/.,;:)`'\"")

    def test_nothing_points_at_01_protocols(self):
        bad = sorted({f"{p}  <- {f}" for f, p in self._hits() if p.startswith("01_PROTOCOLS/")})
        self.assertEqual(bad, [], "\n".join(bad))

    def test_forge_paths_are_real_files_or_declared_script_outputs(self):
        bad = sorted({f"{p}  <- {f}" for f, p in self._hits()
                      if p.startswith("forge/") and not (REPO / p).exists() and p not in self.FORGE_OUTPUTS})
        self.assertEqual(bad, [], "\n".join(bad))

    def test_the_forge_output_workspace_is_gitignored_so_generated_training_data_is_not_committed(self):
        r = subprocess.run(["git", "check-ignore", *self.FORGE_OUTPUTS], cwd=str(REPO), capture_output=True, text=True)
        self.assertEqual(sorted(r.stdout.split()), sorted(self.FORGE_OUTPUTS), r.stdout)


class TestMission052Brief(unittest.TestCase):
    """The orchestration brief is a live instruction for a clean session: every path and mission link in it must resolve."""

    BRIEF = REPO / "LEARNING" / "experiments" / "mission-052" / "brief.md"

    def test_mission_folder_is_complete_and_the_manifest_points_at_it(self):
        import json
        m = json.loads((REPO / "LEARNING" / "missions" / "MISSION-052.json").read_text(encoding="utf-8"))
        self.assertTrue((REPO / m["experiment"]).exists())
        self.assertTrue(self.BRIEF.exists())
        self.assertIn("Results", (REPO / "LEARNING/experiments/mission-052/README.md").read_text(encoding="utf-8"))

    def test_every_mission_link_and_repo_path_in_the_brief_resolves(self):
        text = self.BRIEF.read_text(encoding="utf-8")
        broken = [t for t in re.findall(r"\]\((\.\./\.\./missions/[^)\s]+)\)", text) if not (self.BRIEF.parent / t).exists()]
        for m in REPO_PATH_RX.findall(text):
            p = m.rstrip("/.,;:)`'\"")
            if any(x in p for x in ("NNN", "<", "*", "{")) or p == "LEARNING/research/missions":  # the last is the documented path correction
                continue
            if not (REPO / p).exists():
                broken.append(p)
        self.assertEqual(broken, [], broken)

    def test_the_brief_carries_no_retired_mission_path_and_keeps_the_human_gate(self):
        text = self.BRIEF.read_text(encoding="utf-8")
        self.assertNotIn("LEARNING/research/missions", text.replace("`LEARNING/research/missions/` is now", ""))
        self.assertIn("Do not push, merge into protected `main`", text)

"""Step 1 of the LEARNING restructure: the mission manifests and their schema live in LEARNING/missions/ (not LEARNING/research/missions/)."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
from manage_backlog import BacklogManager  # noqa: E402

REPO = HERE.parents[6]
OLD = ("LEARNING/research/missions", "LEARNING/research/mission.schema.json")
# history and the documents that describe the move are allowed to name the old path
# experiment folders are dated records and recorded data: their paths describe what was true when they ran (only their scripts must keep working)
ALLOWED_PREFIXES = ("chronicle/", "huggingface/", "ARCHIVE/", "LEARNING/experiments/")
ALLOWED_FILES = ("plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/tests/test_mission_paths.py",
                 "plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/manage_backlog.py",  # the stray-file guard names the old path on purpose
                 "plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/tests/test_learning_index.py",  # RETIRED_ON_PURPOSE
                 ".agent/learning/events.jsonl")  # append-only history


class TestMissionPaths(unittest.TestCase):
    def test_manifests_and_schema_are_in_learning_missions(self):
        d = REPO / "LEARNING" / "missions"
        self.assertTrue(list(d.glob("MISSION-*.json")))
        self.assertTrue((d / "mission.schema.json").exists())
        self.assertFalse((REPO / "LEARNING" / "research" / "missions").exists())
        self.assertFalse((REPO / "LEARNING" / "research" / "mission.schema.json").exists())

    def test_no_tracked_file_still_names_the_old_mission_paths(self):
        files = subprocess.run(["git", "grep", "-l", "-F", "-e", OLD[0], "-e", OLD[1]], cwd=str(REPO), capture_output=True, text=True).stdout.split()
        offenders = [f for f in files if not f.startswith(ALLOWED_PREFIXES) and f not in ALLOWED_FILES]
        self.assertEqual(offenders, [], offenders)

    def test_backlog_manager_reads_manifests_from_learning_missions_and_ignores_the_schema(self):
        root = Path(tempfile.mkdtemp())
        mdir = root / "LEARNING" / "missions"; mdir.mkdir(parents=True)
        good = json.loads((REPO / "LEARNING" / "missions" / "MISSION-034.json").read_text())
        (mdir / "MISSION-034.json").write_text(json.dumps(good))
        shutil.copy(REPO / "LEARNING" / "missions" / "mission.schema.json", mdir / "mission.schema.json")
        mgr = BacklogManager(root)
        self.assertEqual(mgr.missions_dir.resolve(), mdir.resolve())
        self.assertEqual([m["id"] for m in mgr.load_missions()], ["MISSION-034"])
        self.assertEqual(mgr.validate(), [])  # the schema file is applied, not mistaken for a mission

    def test_a_manifest_left_at_the_old_path_is_reported_not_silently_ignored(self):
        root = Path(tempfile.mkdtemp())
        (root / "LEARNING" / "missions").mkdir(parents=True)
        old = root / "LEARNING" / "research" / "missions"; old.mkdir(parents=True)
        (old / "MISSION-999.json").write_text("{}")
        errors = BacklogManager(root).validate()
        self.assertTrue(any("LEARNING/research" in e and "missions" in e and "git mv" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()


class TestBacklogViewPath(unittest.TestCase):
    """Step 2: the generated human view moved from LEARNING/research/backlog.md to LEARNING/backlog.md."""

    def test_backlog_view_is_at_the_learning_root(self):
        self.assertTrue((REPO / "LEARNING" / "backlog.md").exists())
        self.assertFalse((REPO / "LEARNING" / "research" / "backlog.md").exists())

    def test_no_tracked_file_still_names_the_old_backlog_path(self):
        files = subprocess.run(["git", "grep", "-l", "-F", "LEARNING/research/backlog.md"], cwd=str(REPO), capture_output=True, text=True).stdout.split()
        offenders = [f for f in files if not f.startswith(ALLOWED_PREFIXES) and f not in ALLOWED_FILES]
        self.assertEqual(offenders, [], offenders)

    def test_compile_writes_the_backlog_view_at_the_learning_root(self):
        root = Path(tempfile.mkdtemp())
        mdir = root / "LEARNING" / "missions"; mdir.mkdir(parents=True)
        (mdir / "MISSION-034.json").write_text((REPO / "LEARNING" / "missions" / "MISSION-034.json").read_text())
        mgr = BacklogManager(root)
        mgr.compile()
        self.assertEqual(mgr.markdown_file.resolve(), (root / "LEARNING" / "backlog.md").resolve())
        self.assertIn("MISSION-034", (root / "LEARNING" / "backlog.md").read_text())
        self.assertFalse((root / "LEARNING" / "research" / "backlog.md").exists())


EXPECTED_ROOT = {"INDEX.md", "README.md", "backlog.md", "index.jsonl", "inbox.jsonl", "concepts", "procedures", "philosophy", "missions", "experiments", "research"}
MISSION_SYSTEM_NAMES = ("missions", "experiments", "index.jsonl", "inbox.jsonl", "backlog.md", "mission.schema.json")
RETIRED = ("LEARNING/research/experiments", "LEARNING/research/index.jsonl", "LEARNING/research/inbox.jsonl", "LEARNING/topics")


class TestFinalLayout(unittest.TestCase):
    """Steps 1-4 done: the mission system is at the LEARNING root and research/ holds the research dossiers (the former topics/)."""

    def test_learning_root_has_exactly_the_expected_entries(self):
        got = {p.name for p in (REPO / "LEARNING").iterdir() if not p.name.startswith(".")}
        self.assertEqual(got, EXPECTED_ROOT)

    def test_research_holds_dossiers_and_none_of_the_mission_system(self):
        research = REPO / "LEARNING" / "research"
        self.assertTrue((research / "README.md").exists())
        self.assertTrue((research / "quantum_error_correction").is_dir())
        self.assertEqual([n for n in MISSION_SYSTEM_NAMES if (research / n).exists()], [])
        self.assertFalse((REPO / "LEARNING" / "topics").exists())

    def test_no_tracked_file_names_a_retired_path(self):
        args = ["git", "grep", "-l", "-F"]
        for r in RETIRED:
            args += ["-e", r]
        files = subprocess.run(args, cwd=str(REPO), capture_output=True, text=True).stdout.split()
        offenders = [f for f in files if not f.startswith(ALLOWED_PREFIXES) and f not in ALLOWED_FILES]
        self.assertEqual(offenders, [], offenders)

    def test_manager_uses_the_root_files_and_compile_writes_the_index_there(self):
        root = Path(tempfile.mkdtemp())
        mdir = root / "LEARNING" / "missions"; mdir.mkdir(parents=True)
        (mdir / "MISSION-034.json").write_text((REPO / "LEARNING" / "missions" / "MISSION-034.json").read_text())
        mgr = BacklogManager(root)
        self.assertEqual(mgr.index_file.resolve(), (root / "LEARNING" / "index.jsonl").resolve())
        self.assertEqual(mgr.inbox_file.resolve(), (root / "LEARNING" / "inbox.jsonl").resolve())
        mgr.compile()
        self.assertTrue((root / "LEARNING" / "index.jsonl").exists())
        self.assertFalse((root / "LEARNING" / "research").exists())

    def test_mission_system_files_left_inside_research_are_reported(self):
        root = Path(tempfile.mkdtemp())
        (root / "LEARNING" / "missions").mkdir(parents=True)
        r = root / "LEARNING" / "research"; r.mkdir(parents=True)
        (r / "index.jsonl").write_text("{}\n"); (r / "inbox.jsonl").write_text("{}\n"); (r / "experiments").mkdir()
        errors = BacklogManager(root).validate()
        self.assertTrue(any("LEARNING/research" in e and "index.jsonl" in e for e in errors), errors)

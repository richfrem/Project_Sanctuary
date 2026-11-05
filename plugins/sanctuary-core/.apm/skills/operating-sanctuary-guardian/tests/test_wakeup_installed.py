"""The wakeup warns when the installed copies of the Sanctuary skills (.claude/skills, .agents/skills) differ from plugins/sanctuary-core."""
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
import guardian_wakeup as gw  # noqa: E402

REPO = HERE.parents[6]
SRC_REL = "plugins/sanctuary-core/.apm/skills"


def _project(install_dirs=(".claude/skills",), drift=False, extra_file=False):
    root = Path(tempfile.mkdtemp())
    for skill in ("alpha", "beta"):
        d = root / SRC_REL / skill / "scripts"; d.mkdir(parents=True)
        (d / "run.py").write_text(f"print('{skill}')\n")
        (root / SRC_REL / skill / "SKILL.md").write_text(f"# {skill}\n")
    for inst in install_dirs:
        shutil.copytree(root / SRC_REL, root / inst)
    if drift:
        (root / install_dirs[0] / "alpha" / "scripts" / "run.py").write_text("print('OLD')\n")
    if extra_file:
        (root / SRC_REL / "beta" / "NEW.md").write_text("new\n")
    return root


def _check(root):
    _, checks = gw.check_layer1_health(root)
    return next((c for c in checks if c["path"] == "installed-skills"), None)


class TestInstalledSkillsDrift(unittest.TestCase):
    def test_identical_copies_pass(self):
        c = _check(_project())
        self.assertEqual(c["status"], "PASS")

    def test_a_changed_script_in_the_installed_copy_warns_and_names_the_fix(self):
        c = _check(_project(drift=True))
        self.assertEqual(c["status"], "WARN")
        self.assertIn("alpha", c["description"])
        self.assertIn("apm install", c["description"])

    def test_a_file_added_to_the_source_but_missing_from_the_install_warns(self):
        c = _check(_project(extra_file=True))
        self.assertEqual(c["status"], "WARN")
        self.assertIn("beta", c["description"])

    def test_every_install_location_is_checked(self):
        c = _check(_project(install_dirs=(".claude/skills", ".agents/skills"), drift=True))
        self.assertEqual(c["status"], "WARN")

    def test_no_install_folder_means_no_check(self):
        self.assertIsNone(_check(_project(install_dirs=())))

    def test_a_missing_source_means_no_check_and_no_crash(self):
        root = Path(tempfile.mkdtemp())
        (root / ".claude" / "skills" / "x").mkdir(parents=True)
        self.assertIsNone(_check(root))

    def test_the_check_is_fast_on_the_real_repo(self):
        t0 = time.perf_counter()
        _check(REPO)
        self.assertLess((time.perf_counter() - t0) * 1000, 300)


if __name__ == "__main__":
    unittest.main()


class TestApmLinkRewriting(unittest.TestCase):
    """APM rewrites relative links between skills on install; that is not drift."""

    def test_a_rewritten_cross_skill_link_is_not_reported_as_drift(self):
        root = _project()
        src_doc = root / SRC_REL / "alpha" / "SKILL.md"
        src_doc.write_text("# alpha\nSee [beta](../beta/SKILL.md) and [ref](../../beta/SKILL.md).\n")
        inst_doc = root / ".claude" / "skills" / "alpha" / "SKILL.md"
        inst_doc.write_text("# alpha\nSee [beta](../../apm_modules/_local/sanctuary-core/.apm/skills/beta/SKILL.md) and "
                            "[ref](../../../../apm_modules/_local/sanctuary-core/.apm/skills/beta/SKILL.md).\n")
        self.assertEqual(_check(root)["status"], "PASS")

    def test_a_real_text_change_next_to_a_rewritten_link_is_still_drift(self):
        root = _project()
        (root / SRC_REL / "alpha" / "SKILL.md").write_text("# alpha\nNEW sentence. [b](../beta/SKILL.md)\n")
        (root / ".claude" / "skills" / "alpha" / "SKILL.md").write_text("# alpha\nOLD sentence. [b](../../apm_modules/_local/sanctuary-core/.apm/skills/beta/SKILL.md)\n")
        self.assertEqual(_check(root)["status"], "WARN")

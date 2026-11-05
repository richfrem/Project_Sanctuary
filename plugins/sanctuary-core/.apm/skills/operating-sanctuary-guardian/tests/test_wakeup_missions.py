"""The wakeup bootloader validates mission manifests against LEARNING/research/mission.schema.json (bootstrapper awareness)."""
import copy
import json
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
KEY = "LEARNING/research/missions"


def _root(with_schema=True, manifests=1):
    root = Path(tempfile.mkdtemp())
    res = root / "LEARNING" / "research"
    (res / "missions").mkdir(parents=True)
    if with_schema:
        shutil.copy(REPO / "LEARNING/research/mission.schema.json", res / "mission.schema.json")
    good = json.loads((REPO / "LEARNING/research/missions/MISSION-034.json").read_text())
    for i in range(manifests):
        m = {**good, "id": f"MISSION-{900 + i}"}
        (res / "missions" / f"MISSION-{900 + i}.json").write_text(json.dumps(m))
    return root


def _check(root):
    _, checks = gw.check_layer1_health(root)
    return next((c for c in checks if c["path"] == KEY), None)


class TestWakeupMissionCheck(unittest.TestCase):
    def test_valid_manifests_pass_and_are_counted(self):
        c = _check(_root(manifests=3))
        self.assertEqual(c["status"], "PASS")
        self.assertIn("3 manifests valid", c["description"])

    def test_invalid_manifest_warns_and_names_the_mission_without_turning_layer1_red(self):
        root = _root(manifests=2)
        p = root / "LEARNING/research/missions/MISSION-900.json"
        m = json.loads(p.read_text()); m["status"] = "DONE"; p.write_text(json.dumps(m))
        health, checks = gw.check_layer1_health(root)
        c = next(x for x in checks if x["path"] == KEY)
        self.assertEqual(c["status"], "WARN")
        self.assertIn("MISSION-900", c["description"])
        self.assertIn("status", c["description"])
        self.assertIn(health, ("GREEN", "YELLOW"))  # the context-file checks decide this, not the manifests

    def test_no_schema_means_no_check(self):
        self.assertIsNone(_check(_root(with_schema=False)))

    def test_schema_but_no_manifests_warns(self):
        c = _check(_root(manifests=0))
        self.assertEqual(c["status"], "WARN")
        self.assertIn("no manifests", c["description"].lower())

    def test_broken_schema_file_warns_instead_of_crashing_the_boot(self):
        root = _root()
        (root / "LEARNING/research/mission.schema.json").write_text("{not json")
        self.assertEqual(_check(root)["status"], "WARN")

    def test_real_repo_check_is_fast_enough_for_the_boot_hook(self):
        t0 = time.perf_counter()
        c = _check(REPO)
        self.assertLess((time.perf_counter() - t0) * 1000, 300)
        self.assertEqual(c["status"], "PASS", c)


if __name__ == "__main__":
    unittest.main()

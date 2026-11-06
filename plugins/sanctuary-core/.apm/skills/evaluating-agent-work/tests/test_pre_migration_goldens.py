"""Compatibility proof for MISSION-055: scoring, cost weights, budget ledger/check output and cost_usd of the REFACTORED scripts equal
the outputs captured from the pre-migration code (fixtures/pre_migration_goldens.json, produced by golden_capture.py on commit 1f464bd52).
Historical scores keep their meaning; any difference here is a breaking change and must be reported, not accepted silently."""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Goldens(unittest.TestCase):
    def test_refactored_scripts_reproduce_pre_migration_outputs(self):
        out = Path(tempfile.mkdtemp()) / "now.json"
        r = subprocess.run([sys.executable, str(HERE / "golden_capture.py"), str(HERE.parent / "scripts"), str(out)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        now = json.loads(out.read_text())
        gold = json.loads((HERE / "fixtures" / "pre_migration_goldens.json").read_text())
        for key in gold:
            self.assertEqual(now[key], gold[key], f"behavior changed for '{key}'")


if __name__ == "__main__":
    unittest.main()

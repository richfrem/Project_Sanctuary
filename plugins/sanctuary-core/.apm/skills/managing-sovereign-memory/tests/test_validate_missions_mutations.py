"""Mutation + default-location tests for validate_missions (MISSION-schema task, written by claude-code on real manifests).

Haiku 5.5 wrote validate_missions.py and its own unit tests; these cover what its spec did not: real manifests, the default
missions folder, and refusing to report OK when nothing was checked."""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import validate_missions as vm  # noqa: E402

REPO = Path(__file__).resolve().parents[6]
MISSIONS = REPO / "LEARNING" / "research" / "missions"
SCHEMA = vm.load_schema()
BASE = json.loads((MISSIONS / "MISSION-034.json").read_text())


def _mutate(fn):
    m = copy.deepcopy(BASE)
    fn(m)
    return vm.validate_manifest(m, SCHEMA)


def test_real_manifest_is_valid():
    assert vm.validate_manifest(BASE, SCHEMA) == []


@pytest.mark.parametrize("name,fn,needle", [
    ("bad id", lambda m: m.__setitem__("id", "MISSION-34"), "id"),
    ("bad status", lambda m: m.__setitem__("status", "DONE"), "status"),
    ("bad verdict", lambda m: m.__setitem__("verdict", "FINISHED"), "verdict"),
    ("missing field", lambda m: m.pop("hypothesis"), "hypothesis"),
    ("wrong type", lambda m: m.__setitem__("tags", "not-a-list"), "tags"),
    ("bad related key", lambda m: m.__setitem__("related_missions", {"mission-31": "x"}), "related_missions"),
    ("bad tier", lambda m: m["model_routing"].__setitem__("cheapest_viable_tier", "premium"), "cheapest_viable_tier"),
    ("score out of range", lambda m: m["scores"].__setitem__("strategic_leverage", 9), "strategic_leverage"),
])
def test_each_mutation_is_rejected_with_its_property_in_the_message(name, fn, needle):
    errors = _mutate(fn)
    assert errors, name
    assert any(needle in e for e in errors), (name, errors)
    assert all(e.startswith("[MISSION-034]") or e.startswith("[") for e in errors)


def test_legacy_confirmed_verdict_is_accepted():
    assert _mutate(lambda m: m.__setitem__("verdict", "CONFIRMED")) == []


def test_unknown_optional_keys_never_break_a_draft():
    assert _mutate(lambda m: m.__setitem__("some_new_field", {"x": 1})) == []


def test_every_tracked_manifest_in_the_repo_validates():
    errors = vm.validate_dir(str(MISSIONS), SCHEMA)
    assert errors == [], errors[:5]


def test_default_missions_dir_is_the_real_folder_not_the_script_folder():
    assert Path(vm.find_missions_dir()).resolve() == MISSIONS.resolve()


def test_zero_manifests_is_an_error_not_ok():
    with tempfile.TemporaryDirectory() as d:
        errors = vm.validate_dir(d, SCHEMA)
    assert errors and "no manifests" in errors[0].lower()


def test_filename_must_match_id():
    with tempfile.TemporaryDirectory() as d:
        m = copy.deepcopy(BASE)
        (Path(d) / "MISSION-099.json").write_text(json.dumps(m))  # contains id MISSION-034
        errors = vm.validate_dir(d, SCHEMA)
    assert any("filename" in e.lower() and "MISSION-099" in e for e in errors), errors


def test_cli_exit_codes(tmp_path):
    import subprocess
    ok = subprocess.run([sys.executable, str(SCRIPTS / "validate_missions.py")], capture_output=True, text=True, cwd=str(tmp_path))
    assert ok.returncode == 0 and ok.stdout.startswith("OK:") and "OK: 0 " not in ok.stdout, ok.stdout
    empty = subprocess.run([sys.executable, str(SCRIPTS / "validate_missions.py"), "--missions-dir", str(tmp_path)], capture_output=True, text=True)
    assert empty.returncode == 1

"""capture_snapshot must find a bundler: the legacy in-repo plugin or the installed bundling-context skill (the legacy path no longer exists)."""
import importlib.util, sys, tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_snapshot.py"
spec = importlib.util.spec_from_file_location("capture_snapshot", SCRIPT); cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text("#"); return p


def test_installed_skill_is_found_when_legacy_plugin_is_absent():
    root, home = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    assert cs.find_bundler(root, home) == (None, None)
    b = _touch(home / ".claude/skills/bundling-context/scripts/bundle.py")
    assert cs.find_bundler(root, home) == (b, "bundling-context")


def test_legacy_plugin_wins_when_present():
    root, home = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    _touch(home / ".agents/skills/bundling-context/scripts/bundle.py")
    leg = _touch(root / "plugins/context-bundler/scripts/bundle.py")
    assert cs.find_bundler(root, home) == (leg, "legacy")

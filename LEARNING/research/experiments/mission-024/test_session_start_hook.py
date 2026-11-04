"""test_session_start_hook.py - Empirical Verification of MISSION-024.

Verifies:
1. Hook schema validity via validate_hook_schema.py and JSON structure
2. Context guard behavior (silent exit 0 outside Project Sanctuary)
3. Layer 1 hydration (.agent/learning/guardian_boot_digest.md persistence)
4. Hook output format (Claude Code hookSpecificOutput schema)
5. Latency threshold (overhead < 1,500ms falsification ceiling)
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
HOOK_SCRIPT = REPO_ROOT / "plugins" / "sanctuary-core" / ".apm" / "hooks" / "session_start_wakeup.py"
HOOK_JSON = REPO_ROOT / "plugins" / "sanctuary-core" / ".apm" / "hooks" / "hooks.json"


def test_hook_files_exist():
    """Verify hook configuration and handler script exist."""
    assert HOOK_JSON.exists(), f"hooks.json missing at {HOOK_JSON}"
    assert HOOK_SCRIPT.exists(), f"Handler script missing at {HOOK_SCRIPT}"
    assert os.access(HOOK_SCRIPT, os.X_OK), "Handler script must be executable"


def test_hook_json_schema():
    """Verify hooks.json conforms to the APM lifecycle hook specification."""
    data = json.loads(HOOK_JSON.read_text(encoding="utf-8"))
    hooks_map = data.get("hooks", data)
    assert "SessionStart" in hooks_map, "SessionStart event must be registered"
    entries = hooks_map["SessionStart"]
    assert len(entries) > 0, "SessionStart entries cannot be empty"

    entry = entries[0]
    assert entry.get("matcher") == ".*"
    handlers = entry.get("hooks", [])
    assert len(handlers) > 0

    handler = handlers[0]
    assert handler.get("type") == "command"
    assert "${CLAUDE_PLUGIN_ROOT}/.apm/hooks/session_start_wakeup.py" in handler.get("command", "")


def test_context_guard_silent_exit():
    """Verify handler exits 0 silently when run outside initialized context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env={"PATH": os.environ.get("PATH", "")}
        )
        assert res.returncode == 0
        assert res.stdout.strip() == ""
        assert res.stderr.strip() == ""


def test_hook_execution_and_digest_persistence():
    """Verify hook execution produces .agent/learning/guardian_boot_digest.md and emits valid JSON."""
    digest_path = REPO_ROOT / ".agent" / "learning" / "guardian_boot_digest.md"
    if digest_path.exists():
        digest_path.unlink()

    env = dict(os.environ)
    env["PROJECT_ROOT"] = str(REPO_ROOT)

    res = subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env
    )

    assert res.returncode == 0, f"Hook failed with stderr: {res.stderr}"
    assert digest_path.exists(), "Hook failed to generate guardian_boot_digest.md"

    # Verify digest content
    digest_content = digest_path.read_text(encoding="utf-8")
    assert "# 🛡️ Guardian Wakeup Briefing" in digest_content
    assert "System Status" in digest_content
    assert "Layer 1 Integrity Health Checks" in digest_content

    # Verify stdout JSON structure
    out_json = json.loads(res.stdout)
    assert "hookSpecificOutput" in out_json
    hook_out = out_json["hookSpecificOutput"]
    assert hook_out.get("hookEventName") == "SessionStart"
    assert "additionalContext" in hook_out
    assert "# 🛡️ Guardian Wakeup Briefing" in hook_out["additionalContext"]


def test_latency_under_falsification_ceiling():
    """Empirically benchmark hook execution latency over 3 trials."""
    env = dict(os.environ)
    env["PROJECT_ROOT"] = str(REPO_ROOT)

    latencies = []
    for _ in range(3):
        t0 = time.perf_counter()
        res = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert res.returncode == 0
        latencies.append(elapsed_ms)

    mean_latency = sum(latencies) / len(latencies)
    print(f"\nEmpirical Latencies: {[f'{x:.1f}ms' for x in latencies]} (Mean: {mean_latency:.1f}ms)")
    # Falsification criterion: total execution overhead exceeds 1,500ms
    assert mean_latency < 1500.0, f"Mean latency {mean_latency:.1f}ms exceeded 1,500ms falsification ceiling"

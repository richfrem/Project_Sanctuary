"""Integration tests for the PreCompact hook and its SessionStart read-back (MISSION-030). Written by claude-code;
test_pre_compact_snapshot.py (unit tests) was written by Codex from the spec."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOOKS = Path(__file__).resolve().parents[1]
HANDLER = HOOKS / "pre_compact_snapshot.py"
WAKEUP = HOOKS / "session_start_wakeup.py"
SECRET = "hf_ABCDEFGHIJKLMNOP1234"


def _project(tmp_path):
    (tmp_path / "context").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "dirty.txt").write_text("x")
    return tmp_path


def _transcript(tmp_path, n=3000):
    lines = []
    for i in range(n):
        lines.append(json.dumps({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": f"line {i} ok"}]}}))
    lines.append(json.dumps({"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "content": "Traceback: boom"}]}}))
    lines.append(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": f"Conclusion: use approach B. token {SECRET}"}]}}))
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(lines))
    return p


def _run(payload, cwd, env_extra=None):
    env = {**os.environ, "PROJECT_ROOT": "", "CLAUDE_PROJECT_DIR": "", **(env_extra or {})}
    return subprocess.run([sys.executable, str(HANDLER)], input=payload, capture_output=True, text=True,
                          cwd=str(cwd), env=env, timeout=10)


def test_end_to_end_snapshot_has_files_conclusion_friction_and_no_secret(tmp_path):
    root = _project(tmp_path)
    t = _transcript(tmp_path)
    r = _run(json.dumps({"session_id": "s1234567", "transcript_path": str(t), "cwd": str(root), "trigger": "auto"}), root)
    assert r.returncode == 0 and r.stdout == ""
    snap = next((root / ".agent/learning/precompact").glob("*.md")).read_text()
    assert "dirty.txt" in snap and "Conclusion: use approach B" in snap and "Traceback: boom" in snap
    assert SECRET not in snap and "[REDACTED]" in snap


def test_latency_under_falsification_ceiling_on_large_transcript(tmp_path):
    root = _project(tmp_path)
    t = _transcript(tmp_path, n=60000)  # several MB
    assert t.stat().st_size > 1_000_000
    payload = json.dumps({"session_id": "s", "transcript_path": str(t), "cwd": str(root)})
    times = []
    for _ in range(5):
        t0 = time.perf_counter(); _run(payload, root); times.append((time.perf_counter() - t0) * 1000)
    assert max(times) < 2500, times


def test_never_blocks_outside_sanctuary_or_with_garbage(tmp_path):
    for payload in ("", "not json", "[]", json.dumps({"transcript_path": "/nope", "cwd": str(tmp_path)})):
        assert _run(payload, tmp_path).returncode == 0
    assert not (tmp_path / ".agent").exists()


def test_session_start_points_at_snapshot_only_after_compaction(tmp_path):
    root = _project(tmp_path)
    _run(json.dumps({"session_id": "s1", "cwd": str(root), "trigger": "manual"}), root)
    env = {**os.environ, "PROJECT_ROOT": str(root)}
    for source, expected in (("compact", True), ("startup", False)):
        r = subprocess.run([sys.executable, str(WAKEUP)], input=json.dumps({"source": source}), capture_output=True,
                           text=True, cwd=str(root), env=env, timeout=30)
        ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"] if r.stdout.strip() else ""
        assert ("Pre-compaction snapshot" in ctx) is expected, (source, r.stderr[-300:])


def test_hooks_json_registers_both_events():
    d = json.loads((HOOKS / "hooks.json").read_text())["hooks"]
    assert {"SessionStart", "PreCompact"} <= set(d)
    cmd = d["PreCompact"][0]["hooks"][0]
    assert cmd["type"] == "command" and "pre_compact_snapshot.py" in cmd["command"]


def test_session_start_injects_the_predecessor_relay_with_the_latest_handoff(tmp_path):
    """The hook builds its own digest; a fresh session must receive the same handoffs the manual wakeup shows."""
    root = _project(tmp_path)
    (root / ".agent" / "learning").mkdir(parents=True)
    (root / ".agent" / "learning" / "events.jsonl").write_text(json.dumps({
        "event_type": "CHECKIN", "agent_id": "codex", "mission_id": "TASK-X", "outcome_status": "COMPLETED", "pr_number": 9,
        "timestamp": "2026-11-05T01:00:00+00:00", "handoff": "ZETA-handoff: do not redo the schema; next step is Y"}) + "\n")
    env = {**os.environ, "PROJECT_ROOT": str(root)}
    r = subprocess.run([sys.executable, str(WAKEUP)], input=json.dumps({"source": "startup"}), capture_output=True, text=True,
                       cwd=str(root), env=env, timeout=30)
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Predecessor Relay" in ctx and "ZETA-handoff" in ctx, ctx[-600:]


def test_session_start_still_works_when_the_relay_cannot_be_built(tmp_path):
    root = _project(tmp_path)
    (root / ".agent" / "learning").mkdir(parents=True)
    (root / ".agent" / "learning" / "events.jsonl").write_text("not json at all\n")
    env = {**os.environ, "PROJECT_ROOT": str(root)}
    r = subprocess.run([sys.executable, str(WAKEUP)], input=json.dumps({"source": "startup"}), capture_output=True, text=True,
                       cwd=str(root), env=env, timeout=30)
    assert r.returncode == 0 and "hookSpecificOutput" in r.stdout


def test_hook_commands_try_the_plugin_root_before_any_working_directory_path():
    """A cwd-relative script in an untrusted checkout must never run while the installed plugin copy is available."""
    d = json.loads((HOOKS / "hooks.json").read_text())["hooks"]
    for event in ("SessionStart", "PreCompact"):
        cmd = d[event][0]["hooks"][0]["command"]
        assert cmd.index("${CLAUDE_PLUGIN_ROOT}") < cmd.index('"hooks/sanctuary-core'), cmd

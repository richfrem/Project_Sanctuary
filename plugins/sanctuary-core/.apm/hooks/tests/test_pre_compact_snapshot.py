import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.environ.get("HOOK_DIR", str(Path(__file__).resolve().parents[1])))
import pre_compact_snapshot as m


def test_scrub_redacts_secret_tokens_and_bearer_tokens():
    text = (
        "hf_ABCDEFGHIJKLMNOP1234 "
        "sk-abcdefghijklmnop "
        "ghp_abcdefghijklmnop "
        "Bearer abcdefghijklmnop"
    )

    scrubbed = m.scrub(text)

    assert scrubbed.count("[REDACTED]") == 4
    assert "hf_ABCDEFGHIJKLMNOP1234" not in scrubbed
    assert "sk-abcdefghijklmnop" not in scrubbed
    assert "ghp_abcdefghijklmnop" not in scrubbed
    assert "Bearer abcdefghijklmnop" not in scrubbed


def test_scrub_leaves_normal_text_unchanged():
    text = "A normal transcript line with no credentials."

    assert m.scrub(text) == text


def test_read_transcript_returns_conclusions_and_friction(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    entries = [
        {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "First conclusion."}],
            }
        },
        {
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "tool error: command failed"}],
                    }
                ],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Final conclusion."}],
            }
        },
    ]
    transcript.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    conclusions, friction = m.read_transcript(str(transcript))

    assert conclusions == ["First conclusion.", "Final conclusion."]
    assert friction == ["tool error: command failed"]


def test_read_transcript_missing_path_returns_empty_lists(tmp_path):
    assert m.read_transcript(str(tmp_path / "missing.jsonl")) == ([], [])


def test_find_project_root_returns_directory_containing_context(tmp_path):
    root = tmp_path / "project"
    (root / "context").mkdir(parents=True)

    assert m.find_project_root(str(root)) == root


def test_find_project_root_returns_none_without_context(tmp_path, monkeypatch):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.chdir(unrelated)

    assert m.find_project_root() is None


def test_persist_writes_snapshot_and_latest_and_keeps_at_most_20(tmp_path):
    payload = {"session_id": "session-123", "trigger": "manual"}

    for index in range(22):
        m.persist(tmp_path, f"snapshot {index}", payload)

    snapshot_dir = tmp_path / ".agent" / "learning" / "precompact"
    snapshots = list(snapshot_dir.glob("*.md"))
    latest = json.loads((snapshot_dir / "latest.json").read_text(encoding="utf-8"))

    assert len(snapshots) <= 20
    assert latest["session_id"] == "session-123"
    assert latest["trigger"] == "manual"
    latest_path = snapshot_dir / latest["file"]
    assert latest_path.is_file()
    assert latest_path.read_text(encoding="utf-8").startswith("snapshot ")
    assert all(snapshot.parent == snapshot_dir for snapshot in snapshots)


@pytest.mark.parametrize("stdin_text", ["{malformed", "", "{}"])
def test_main_handles_empty_and_invalid_input_without_context(
    tmp_path, monkeypatch, stdin_text
):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.chdir(unrelated)
    monkeypatch.setattr(m.sys, "stdin", __import__("io").StringIO(stdin_text))

    assert m.main() == 0


def test_planted_transcript_secret_never_appears_in_snapshot(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    secret = "hf_ABCDEFGHIJKLMNOP1234"
    transcript.write_text(
        json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"Conclusion includes {secret}."}],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot = m.build_snapshot(
        {"transcript_path": str(transcript), "trigger": "manual"},
        tmp_path,
        0.0,
    )

    assert secret not in snapshot


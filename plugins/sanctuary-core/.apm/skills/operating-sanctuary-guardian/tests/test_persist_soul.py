"""Tests for persist_soul.py (soul snapshot upload via the global managing-huggingface-assets skill).

Purpose: guard against the former dependency on the removed plugins/huggingface-utils plugin.
Key Input Dependencies: pytest; scripts/persist_soul.py. Uses fake hf_config/hf_upload modules; no network.
Key Functions: test_finds_skill_in_standard_dirs, test_override_env_wins, test_project_root_accepts_git_file,
    test_full_sync_prints_commands, test_missing_skill_is_a_clear_error, test_dry_run, test_upload_success, test_upload_failure_exit
Usage: python3 -m pytest plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests/test_persist_soul.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import persist_soul as ps  # noqa: E402

FAKE_CONFIG = '''
from types import SimpleNamespace
def get_hf_config():
    return SimpleNamespace(dataset_repo_id="user/dataset")
'''
FAKE_UPLOAD = '''
from types import SimpleNamespace
async def upload_soul_snapshot(path, valence, config):
    return SimpleNamespace(success=%s, repo_url="https://hf/x", remote_path="lineage/s.md", error=None)
async def upload_semantic_cache(path, config):
    return SimpleNamespace(success=True, repo_url="u", remote_path="cache", error=None)
'''


@pytest.fixture
def fake_skill(tmp_path, monkeypatch):
    def make(ok=True):
        scripts = tmp_path / "skill" / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "hf_config.py").write_text(FAKE_CONFIG)
        (scripts / "hf_upload.py").write_text(FAKE_UPLOAD % ("True" if ok else "False"))
        for name in ("hf_config", "hf_upload"):
            sys.modules.pop(name, None)
        monkeypatch.setenv("MANAGING_HF_SKILL_PATH", str(scripts))
        return scripts
    yield make
    for name in ("hf_config", "hf_upload"):
        sys.modules.pop(name, None)


def test_finds_skill_in_standard_dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("MANAGING_HF_SKILL_PATH", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert ps.find_managing_hf_skill() is None
    scripts = tmp_path / ".agents" / "skills" / "managing-huggingface-assets" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "hf_config.py").write_text("")
    assert ps.find_managing_hf_skill() == scripts


def test_override_env_wins(tmp_path, monkeypatch):
    scripts = tmp_path / "custom"
    scripts.mkdir()
    (scripts / "hf_config.py").write_text("")
    monkeypatch.setenv("MANAGING_HF_SKILL_PATH", str(scripts))
    assert ps.find_managing_hf_skill() == scripts


def test_project_root_accepts_git_file(tmp_path):
    (tmp_path / ".git").write_text("gitdir: elsewhere")  # a worktree has a .git *file*
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert ps._find_project_root(deep) == tmp_path


def test_full_sync_prints_commands(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["persist_soul.py", "--full-sync"])
    ps.main()
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "redirect" and any("sync_soul_dataset.py" in c for c in out["commands"])
    assert not any("obsidian-integration" in c or "huggingface-utils" in c for c in out["commands"])


def test_missing_skill_is_a_clear_error(monkeypatch, capsys):
    monkeypatch.setattr(ps, "find_managing_hf_skill", lambda: None)
    monkeypatch.setattr(sys, "argv", ["persist_soul.py"])
    with pytest.raises(SystemExit) as exc:
        ps.main()
    assert exc.value.code == 1
    err = json.loads(capsys.readouterr().err)
    assert "managing-huggingface-assets" in err["error"] and "MANAGING_HF_SKILL_PATH" in err["fix"]


def test_dry_run(fake_skill, monkeypatch, capsys):
    fake_skill()
    monkeypatch.setattr(sys, "argv", ["persist_soul.py", "--dry-run", "--snapshot", "AGENTS.md"])
    ps.main()
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "dry-run" and out["dataset"] == "user/dataset"


def test_upload_success(fake_skill, monkeypatch, capsys):
    fake_skill(ok=True)
    monkeypatch.setattr(sys, "argv", ["persist_soul.py", "--snapshot", "AGENTS.md"])
    ps.main()
    assert '"status": "success"' in capsys.readouterr().out


def test_upload_failure_exit(fake_skill, monkeypatch):
    fake_skill(ok=False)
    monkeypatch.setattr(sys, "argv", ["persist_soul.py", "--snapshot", "AGENTS.md"])
    with pytest.raises(SystemExit) as exc:
        ps.main()
    assert exc.value.code == 1

"""Tests for sync_soul_dataset runtime preflight (.env loading without python-dotenv)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sync_soul_dataset as S  # noqa: E402


def test_repo_root_is_repository():
    assert (S.REPO_ROOT / "AGENTS.md").exists()


def test_load_repo_env_sets_missing_keys_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# c\nHF_T_A=one\nexport HF_T_B="two"\nHF_T_C=\nHF_T_D=keep-new\nnot a pair\n')
    monkeypatch.delenv("HF_T_A", raising=False)
    monkeypatch.delenv("HF_T_B", raising=False)
    monkeypatch.setenv("HF_T_D", "existing")
    loaded = S.load_repo_env(env)
    assert os.environ["HF_T_A"] == "one"
    assert os.environ["HF_T_B"] == "two"
    assert os.environ["HF_T_D"] == "existing"
    assert "HF_T_C" not in os.environ
    assert sorted(loaded) == ["HF_T_A", "HF_T_B"]


def test_load_repo_env_missing_file(tmp_path):
    assert S.load_repo_env(tmp_path / "nope.env") == []

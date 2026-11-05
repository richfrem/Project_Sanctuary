"""Hook tests must not depend on the shell they run in: Claude Code and a sourced .env both export PROJECT_ROOT / CLAUDE_PROJECT_DIR."""
import pytest


@pytest.fixture(autouse=True)
def _no_ambient_project_root(monkeypatch):
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)

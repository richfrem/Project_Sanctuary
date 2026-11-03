"""Tests for protocol_manager.py status updates (in-place, content-preserving).

Purpose: guard against the former behaviour where `update` re-rendered the whole file from a template and lost fields.
Key Input Dependencies: pytest; scripts/protocol_manager.py.
Key Functions: test_status_only_changes_the_status_line, test_inline_status_form, test_bullet_form_without_rule,
    test_missing_status_line_is_inserted, test_reason_adds_note, test_update_protocol_end_to_end, test_unknown_status_warns
Usage: python3 -m pytest plugins/sanctuary-core/.apm/skills/managing-protocols/tests/test_protocol_manager.py
"""
import difflib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import protocol_manager as pm  # noqa: E402

TEMPLATE_DOC = """# Protocol 7: Example (v1.0)

**Status:** PROPOSED
**Classification:** Operational
**Version:** 1.0
**Authority:** Someone
**Created:** 2026-01-01
**References:** ADR 1

---

## Body

Text with a table:

| a | b |
|---|---|
| 1 | 2 |

---

Trailing section.
"""


def changed_lines(a: str, b: str):
    return [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]


def test_status_only_changes_the_status_line():
    out = pm._set_status(TEMPLATE_DOC, "CANONICAL", today="2026-11-03")
    assert changed_lines(TEMPLATE_DOC, out) == ["-**Status:** PROPOSED", "+**Status:** CANONICAL"]
    assert "**Created:** 2026-01-01" in out and "Trailing section." in out


def test_inline_status_form():
    doc = "# Protocol 9: X\n\n**Status:** Canonical | **Protocol Class:** Security | **Version:** v1.0  \n\nBody\n"
    out = pm._set_status(doc, "DEPRECATED")
    assert "**Status:** DEPRECATED | **Protocol Class:** Security | **Version:** v1.0" in out


def test_bullet_form_without_rule():
    doc = "# Protocol 94: Y\n*   **Status:** Canonical, Active\n*   **Classification:** Foundational\n\n## 1. Preamble\ntext\n"
    out = pm._set_status(doc, "DEPRECATED")
    assert out.splitlines()[1] == "*   **Status:** DEPRECATED"
    assert "## 1. Preamble" in out and out.endswith("text\n")


def test_missing_status_line_is_inserted():
    doc = "# Protocol 5: Z\n\nSome intro.\n"
    out = pm._set_status(doc, "PROPOSED")
    assert "**Status:** PROPOSED" in out and "Some intro." in out and out.startswith("# Protocol 5: Z")


def test_reason_adds_note():
    out = pm._set_status(TEMPLATE_DOC, "DEPRECATED", reason="Superseded by 128.", today="2026-11-03")
    assert "> **Status change (2026-11-03):** PROPOSED -> DEPRECATED. Superseded by 128." in out
    assert len(changed_lines(TEMPLATE_DOC, out)) == 3


def test_update_protocol_end_to_end(tmp_path, monkeypatch, capsys):
    (tmp_path / "7_Example.md").write_text(TEMPLATE_DOC)
    monkeypatch.setattr(pm, "PROTOCOL_DIR", tmp_path)
    pm.update_protocol(7, "canonical", "Approved")
    text = (tmp_path / "7_Example.md").read_text()
    assert "**Status:** CANONICAL" in text and "| 1 | 2 |" in text and "Trailing section." in text
    assert "Updated Protocol 7" in capsys.readouterr().out


def test_unknown_status_warns(tmp_path, monkeypatch, capsys):
    (tmp_path / "7_Example.md").write_text(TEMPLATE_DOC)
    monkeypatch.setattr(pm, "PROTOCOL_DIR", tmp_path)
    pm.update_protocol(7, "weird")
    assert "not a standard status" in capsys.readouterr().out


def test_missing_protocol_and_no_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pm, "PROTOCOL_DIR", tmp_path)
    pm.update_protocol(99, "CANONICAL")
    assert "not found" in capsys.readouterr().out

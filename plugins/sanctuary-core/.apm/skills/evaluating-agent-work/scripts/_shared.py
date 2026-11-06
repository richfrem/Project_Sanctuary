"""Locate the shared companion skill `evaluating-agent-outcomes` (single implementation of generic evaluation, scoring,
accounting and usage measurement) and put its scripts directory on sys.path. Fails loudly if it is not installed.

Search order: env SHARED_EVAL_SCRIPTS, project .claude/.agents skills, ~/.claude, ~/.agents, sibling companion clone.
`resolution()` reports which path was chosen (used by verification and docs)."""
import os, sys
from pathlib import Path

_REL = Path("evaluating-agent-outcomes") / "scripts"


def candidates():
    c = []
    if os.environ.get("SHARED_EVAL_SCRIPTS"):
        c.append(Path(os.environ["SHARED_EVAL_SCRIPTS"]))
    here = Path(__file__).resolve()
    for up in here.parents:
        if (up / ".git").exists():
            c += [up / ".claude/skills" / _REL, up / ".agents/skills" / _REL]
            break
    c += [Path.home() / ".claude/skills" / _REL, Path.home() / ".agents/skills" / _REL,
          Path.home() / "Projects/agent-plugins-skills/plugins/agent-orchestration/.apm/skills" / _REL]
    return c


def locate():
    for c in candidates():
        if (c / "eval_core.py").is_file():
            return c
    raise SystemExit("shared skill 'evaluating-agent-outcomes' not found; run `apm update --global` (companion agent-orchestration plugin) "
                     "or set SHARED_EVAL_SCRIPTS")


SHARED = locate()
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))


def resolution():
    return {"shared_scripts": str(SHARED), "candidates_checked_in_order": [str(c) for c in candidates()]}

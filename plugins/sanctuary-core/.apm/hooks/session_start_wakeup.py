#!/usr/bin/env python3
"""session_start_wakeup.py - Protocol 128 Fast-Boot Lifecycle Hook Handler (SessionStart).

Purpose:
    Executes on SessionStart across all supported agent harnesses (Claude Code, Copilot, Codex, Antigravity).
    Invokes the holistic Protocol 128 bootloader to generate and persist the Layer 1 boot digest,
    injecting zero-turn situational awareness into the incoming agent's session context without manual commands.

Layer: APM Lifecycle Hooks (SF-11 Harness-Neutral).
Key Invariants:
    - Silent project guard: exits 0 immediately if the repository context is uninitialized.
    - Harness-neutral state: never references machine-local harness paths for project state.
    - Deterministic output: emits Claude Code hookSpecificOutput JSON on stdout for context injection.
    - Latency optimization: imports bootloader in-process to eliminate secondary interpreter spawn overhead.
"""

import os
import sys
import time
import json
import importlib.util
from pathlib import Path

# Measure hook overhead from process start
HOOK_START_TIME = time.perf_counter()

HERE = Path(__file__).resolve().parent


def find_project_root() -> Path | None:
    """Resolve project root using environment variables or active working directory."""
    if os.environ.get("PROJECT_ROOT"):
        p = Path(os.environ["PROJECT_ROOT"]).resolve()
        if (p / "context").is_dir():
            return p

    if os.environ.get("CLAUDE_PROJECT_DIR"):
        p = Path(os.environ["CLAUDE_PROJECT_DIR"]).resolve()
        if (p / "context").is_dir():
            return p

    # Check current working directory
    cwd = Path.cwd().resolve()
    if (cwd / "context").is_dir():
        return cwd

    # Check parents of cwd in case the agent started in a repository subfolder
    for parent in cwd.parents:
        if (parent / "context").is_dir():
            return parent

    return None


def execute_bootloader_in_process(project_root: Path) -> tuple[int, str, float]:
    """Execute guardian wakeup logic in-process to minimize latency overhead."""
    t0 = time.perf_counter()
    guardian_script = (
        HERE.parent
        / "skills"
        / "operating-sanctuary-guardian"
        / "scripts"
        / "guardian_wakeup.py"
    )
    if not guardian_script.exists():
        guardian_script = (
            project_root
            / "plugins"
            / "sanctuary-core"
            / ".apm"
            / "skills"
            / "operating-sanctuary-guardian"
            / "scripts"
            / "guardian_wakeup.py"
        )

    if not guardian_script.exists():
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return 1, "guardian_wakeup.py not found", elapsed_ms

    try:
        # Dynamically import guardian_wakeup module
        spec = importlib.util.spec_from_file_location("guardian_wakeup", str(guardian_script))
        if spec is None or spec.loader is None:
            raise ImportError("Could not load spec for guardian_wakeup")
        gw = importlib.util.module_from_spec(spec)
        sys.modules["guardian_wakeup"] = gw
        spec.loader.exec_module(gw)

        # Run core diagnostics
        git_meta = gw.get_git_telemetry(project_root)
        health_status, checks = gw.check_layer1_health(project_root)
        runtime_checks = gw.check_runtime_telemetry()
        top_mission = gw.get_top_mission(project_root)

        digest_lines = [
            "# 🛡️ Guardian Wakeup Briefing (Protocol 128 Holistic Wakeup)",
            f"**System Status:** {health_status} - Layer 1 Fast-Boot Hydrated",
            f"**Git Environment:** Branch `{git_meta['branch']}` ({'DIRTY' if git_meta['dirty'] else 'CLEAN'})",
            f"**Generated Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC",
            "",
            "## I. Strategic Directives",
            "* **Core Mandate:** Anti-Asch epistemic platform, sovereign memory architecture, and pair-programming partnership with Richard.",
            "* **Supreme Law:** Never perform unapproved state-changing operations or git pushes.",
            "* **Cross-Model Cognitive Relay:** Operating on shared sovereign memory across Antigravity (Gemini), Claude Code (Anthropic), Codex (OpenAI), and Local MLX (Qwen2.5). Check the Flight Board and build upon predecessor work without amnesiac repetition.",
            "",
            "## II. Tactical Priorities",
            gw.get_tactical_priorities(project_root, top_mission),
            "",
            "## III. Pre-Flight Brief (Sovereign Memory Engine)",
            gw.get_preflight_brief(project_root),
            "",
            gw.get_relay_section(project_root),  # Predecessor Relay + fuel gauge: the same handoffs the manual wakeup prints (empty string if it cannot be built)
            "",
            "## IV. Layer 1 Integrity Health Checks",
        ]

        for check in checks:
            digest_lines.append(f"- [{check['status']}] `{check['path']}`: {check['description']} ({check['size_bytes']} bytes)")

        digest_lines.append("")
        digest_lines.append("## V. Runtime & Sovereign Inference Capabilities")
        for rc in runtime_checks:
            digest_lines.append(f"- [{rc['status']}] **{rc['component']}**: {rc['description']}")

        digest_text = "\n".join(digest_lines)

        # Persist digest to disk (.agent/learning/guardian_boot_digest.md)
        learning_dir = project_root / ".agent" / "learning"
        learning_dir.mkdir(parents=True, exist_ok=True)
        digest_path = learning_dir / "guardian_boot_digest.md"
        with open(digest_path, "w", encoding="utf-8") as f:
            f.write(digest_text)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return 0, digest_text, elapsed_ms

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return 1, f"In-process execution failed: {e}", elapsed_ms


def precompact_pointer(project_root: Path, raw_payload: str) -> str:
    """After a compaction (SessionStart source == "compact"), point at the snapshot taken by pre_compact_snapshot.py."""
    try:
        if json.loads(raw_payload or "{}").get("source") != "compact":
            return ""
        d = project_root / ".agent" / "learning" / "precompact"
        latest = json.loads((d / "latest.json").read_text(encoding="utf-8"))
        if (d / latest["file"]).exists():
            return (f"\n\n## Pre-compaction snapshot\nContext was just compacted. A provisional snapshot of uncommitted work, "
                    f"recent conclusions and friction was saved to `.agent/learning/precompact/{latest['file']}` - read it first.")
    except Exception:
        pass
    return ""


def main() -> int:
    """Run the SessionStart hook handler."""
    # Read stdin payload if provided by the harness
    raw_payload = ""
    try:
        if not sys.stdin.isatty():
            raw_payload = sys.stdin.read()
    except Exception:
        pass

    # Project Context Guard (SF-11)
    project_root = find_project_root()
    if not project_root:
        # Repository uninitialized or outside Project Sanctuary context; exit silently
        return 0

    code, digest_text, wakeup_elapsed_ms = execute_bootloader_in_process(project_root)

    total_hook_ms = (time.perf_counter() - HOOK_START_TIME) * 1000

    context_banner = (
        f"<!-- Protocol 128 Fast-Boot Hook: Zero-Turn Context Hydration -->\n"
        f"<!-- Total Hook Overhead: {total_hook_ms:.1f}ms (In-Process Wakeup: {wakeup_elapsed_ms:.1f}ms) -->\n\n"
        f"{digest_text}"
        f"{precompact_pointer(project_root, raw_payload)}"
    )

    output_payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context_banner,
        }
    }

    # Emit JSON response on stdout for Claude Code hook ingestion
    print(json.dumps(output_payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())

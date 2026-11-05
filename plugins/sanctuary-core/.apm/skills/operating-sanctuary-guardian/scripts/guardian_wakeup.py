#!/usr/bin/env python3
"""guardian_wakeup.py - Protocol 128 Holistic Wakeup Bootloader.
Purpose: Deterministic situational awareness, cognitive hydration, and runtime telemetry for autonomous AI agents.
Layer: sanctuary-core / operating-sanctuary-guardian (Layer 1 Fast-Boot & Readiness).
Key Inputs: Runtime context files (soul.md, user.md, memory.md), index.jsonl, git telemetry, local MLX server.
Key Outputs: JSON situational awareness digest, human markdown brief, and health metrics (boot time < 10ms).
Usage:
    python3 guardian_wakeup.py --mode HOLISTIC
    python3 guardian_wakeup.py --mode HOLISTIC --write-digest
    python3 guardian_wakeup.py --format markdown
Key Functions: get_git_telemetry(), check_runtime_telemetry(), get_top_mission(), main().
"""

import os
import sys
import time

# Start wall-clock measurement immediately at top of process
PROCESS_START_TIME = time.perf_counter()

import json
import logging
import argparse
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger("guardian_wakeup")


def get_project_root() -> Path:
    if os.environ.get("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def get_git_telemetry(project_root: Path) -> dict[str, str | bool]:
    """Retrieve actual git branch and status dynamically."""
    branch = "unknown"
    dirty = False
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2
        )
        if res.returncode == 0:
            branch = res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2
        )
        if status_res.returncode == 0:
            dirty = bool(status_res.stdout.strip())
    except Exception:
        pass

    return {"branch": branch, "dirty": dirty}


def check_mission_manifests(project_root: Path) -> "dict | None":
    """Validate LEARNING/missions/*.json against LEARNING/missions/mission.schema.json (MISSION schema task).

    None when the project has no schema (older checkouts). Never raises: a problem is reported as WARN so it cannot break the boot.
    Uses the zero-dependency validator of the managing-sovereign-memory skill.
    """
    schema_path = project_root / "LEARNING" / "missions" / "mission.schema.json"
    if not schema_path.exists():
        return None
    entry = {"path": "LEARNING/missions", "status": "WARN", "size_bytes": 0, "description": ""}
    try:
        import importlib.util
        script = Path(__file__).resolve().parents[2] / "managing-sovereign-memory" / "scripts" / "validate_missions.py"
        spec = importlib.util.spec_from_file_location("validate_missions", str(script))
        vm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vm)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        missions_dir = project_root / "LEARNING" / "missions"
        errors = vm.validate_dir(str(missions_dir), schema) if missions_dir.is_dir() else ["[(none)] (root): no manifests found"]
        count = len(list(missions_dir.glob("MISSION-*.json"))) if missions_dir.is_dir() else 0
        if not errors:
            entry.update(status="PASS", description=f"Mission manifests vs mission.schema.json [{count} manifests valid]")
        else:
            shown = "; ".join(errors[:3]) + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else "")
            entry["description"] = f"Mission manifests vs mission.schema.json [{len(errors)} problem(s): {shown}]"
    except Exception as e:  # noqa: BLE001 - boot must not fail on a validator problem
        entry["description"] = f"Mission schema check could not run [{e.__class__.__name__}: {e}]"
    return entry


def check_installed_skills(project_root: Path) -> "dict | None":
    """Compare the installed copies of the Sanctuary skills (.claude/skills, .agents/skills) with plugins/sanctuary-core.

    APM installs real copies, so editing the plugin leaves them stale until `apm install` is run again, and agents then run old scripts.
    None when there is no source or no install folder. Never raises: a problem is reported as WARN so it cannot break the boot.
    """
    import hashlib
    src = project_root / "plugins" / "sanctuary-core" / ".apm" / "skills"
    if not src.is_dir():
        return None
    installs = [project_root / d for d in (".claude/skills", ".agents/skills") if (project_root / d).is_dir()]
    if not installs:
        return None
    entry = {"path": "installed-skills", "status": "WARN", "size_bytes": 0, "description": ""}
    skip = ("__pycache__", ".pyc", ".DS_Store")

    import re
    names = sorted(d.name for d in src.iterdir() if d.is_dir())
    # APM rewrites relative links between skills on install ("../beta/x.md" becomes "../../apm_modules/_local/sanctuary-core/.apm/skills/beta/x.md"):
    # map both forms to one neutral prefix before hashing so that is not reported as drift, while any real text change still is.
    rewritten = re.compile(r"(?:\.\./)+apm_modules/_local/sanctuary-core/\.apm/skills/")
    sibling = re.compile(r"(?:\.\./)+(?=(?:%s)/)" % "|".join(re.escape(n) for n in names)) if names else None

    def digest(f: Path) -> str:
        data = f.read_bytes()
        if f.suffix == ".md":
            text = rewritten.sub("<SKILLS>/", data.decode("utf-8", errors="replace"))
            if sibling:
                text = sibling.sub("<SKILLS>/", text)
            data = text.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def files(base: Path) -> dict:
        out = {}
        for f in base.rglob("*"):
            if f.is_file() and not any(x in str(f) for x in skip):
                out[str(f.relative_to(base))] = digest(f)
        return out

    try:
        drift = []
        skills = sorted(d.name for d in src.iterdir() if d.is_dir())
        for inst in installs:
            for name in skills:
                a = files(src / name)
                b = files(inst / name) if (inst / name).is_dir() else None
                if b is None:
                    drift.append(f"{inst.parent.name}: {name} (not installed)")
                    continue
                n = len([k for k in set(a) | set(b) if a.get(k) != b.get(k)])
                if n:
                    drift.append(f"{inst.parent.name}: {name} ({n} file(s))")
        where = ", ".join(i.parent.name for i in installs)
        if drift:
            shown = "; ".join(drift[:4]) + (f" (+{len(drift) - 4} more)" if len(drift) > 4 else "")
            entry["description"] = f"Installed skills differ from plugins/sanctuary-core [{shown}]. Run `apm install` at the repo root."
        else:
            entry.update(status="PASS", description=f"Installed skills match plugins/sanctuary-core [{len(skills)} skills in {where}]")
    except Exception as e:  # noqa: BLE001 - boot must not fail on a drift-check problem
        entry["description"] = f"Installed-skills check could not run [{e.__class__.__name__}: {e}]"
    return entry


def check_layer1_health(project_root: Path) -> tuple[str, list[dict[str, str]]]:
    """Inspect presence and structural integrity of Layer 1 fast-boot context files."""
    checks = []
    required_files = [
        ("context/soul.md", "Soul Identity & Anti-Asch Principles", "markdown"),
        ("context/user.md", "Human Partner Preferences & Conventions", "markdown"),
        ("context/memory.md", "Curated Long-Term Memory & Active Decisions", "markdown"),
        ("chronicle/ledger.json", "Episodic Chronicle Index", "json")
    ]

    all_passed = True
    for rel_path, desc, ftype in required_files:
        p = project_root / rel_path
        exists = p.exists() and p.is_file()
        size = p.stat().st_size if exists else 0
        valid = False
        detail = ""

        if exists and size > 0:
            try:
                content = p.read_text(encoding="utf-8")
                if ftype == "json":
                    data = json.loads(content)
                    valid = isinstance(data, (dict, list))
                    detail = f"{len(data)} entries" if isinstance(data, (dict, list)) else "invalid shape"
                elif ftype == "markdown":
                    valid = any(line.strip().startswith("#") for line in content.splitlines())
                    detail = "valid headers" if valid else "missing headers"
            except Exception as e:
                valid = False
                detail = f"corrupt ({e.__class__.__name__})"
        else:
            detail = "missing or empty"

        status = "PASS" if valid else "FAIL"
        checks.append({
            "path": rel_path,
            "description": f"{desc} [{detail}]",
            "status": status,
            "size_bytes": size
        })
        if not valid:
            all_passed = False

    mission_check = check_mission_manifests(project_root)
    if mission_check:
        checks.append(mission_check)  # informational: does not decide GREEN/YELLOW (that is the context files)
    installed_check = check_installed_skills(project_root)
    if installed_check:
        checks.append(installed_check)  # informational as well

    health_status = "GREEN" if all_passed else "YELLOW"
    return health_status, checks


def check_runtime_telemetry() -> list[dict[str, str]]:
    """Verify runtime environment: APM CLI, Python/uv execution harness, and local sovereign models."""
    import shutil
    runtime_checks = []

    # 1. APM CLI verification
    apm_path = shutil.which("apm")
    runtime_checks.append({
        "component": "APM Package Manager",
        "status": "PASS" if apm_path else "WARN",
        "description": f"CLI available ({apm_path})" if apm_path else "apm CLI not found in PATH"
    })

    # 2. Local execution harness (uv / uvx)
    uvx_path = shutil.which("uvx") or shutil.which("uv")
    runtime_checks.append({
        "component": "Local MLX Runner (uv/uvx)",
        "status": "PASS" if uvx_path else "WARN",
        "description": f"Runner available ({uvx_path})" if uvx_path else "uv/uvx not found"
    })

    # 3. Cached sovereign model
    hf_model_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit"
    model_cached = hf_model_dir.exists()
    runtime_checks.append({
        "component": "Local Model Weights (Qwen2.5-Coder-7B-4bit)",
        "status": "PASS" if model_cached else "WARN",
        "description": "Cached in ~/.cache/huggingface/hub (offline ready)" if model_cached else "Model not yet cached locally"
    })

    # 4. Local MLX Server daemon. Protocol 128 boot timer: a server this manager started in an earlier session and that has been idle past its
    #    timeout is reaped; a server the manager did not start (an attached or hand-launched persistent daemon) is never touched.
    reap_note = ""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import mlx_server_manager as msm
        reaped = msm.reap_if_idle(only_managed=True)
        if reaped.get("status") == "reaped":
            reap_note = f" Reaped an idle managed server at boot (idle {reaped.get('idle_sec', 0):.0f}s, ~{reaped.get('reclaimed_rss_mb', 0):.0f} MB freed)."
    except Exception:
        pass
    server_live = False
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=0.5) as r:
            server_live = (r.status == 200)
    except Exception:
        pass
    runtime_checks.append({
        "component": "Local MLX Server Daemon (:8080)",
        "status": "ACTIVE" if server_live else "STANDBY",
        "description": ("Responding at http://127.0.0.1:8080/v1 (zero-latency ready)" if server_live else "Standby (auto-starts on demand via forge_llm.py)") + reap_note
    })

    return runtime_checks


def get_top_mission(project_root: Path) -> dict | None:
    """Retrieves highest-priority incomplete research mission from index.jsonl."""
    index_file = project_root / "LEARNING" / "index.jsonl"
    if not index_file.exists():
        return None
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                status = str(data.get("status", "")).upper()
                if status not in ("RESOLVED", "ARCHIVED"):
                    return data
    except Exception:
        pass
    return None


def get_tactical_priorities(project_root: Path, top_mission: dict | None = None) -> str:
    parts = []
    board_file = project_root / ".agent" / "coordination" / "board.json"
    try:  # untracked runtime state lives in the main checkout; resolve it from a worktree too
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import manage_coordination
        board_file = manage_coordination.shared_root(project_root) / ".agent" / "coordination" / "board.json"
    except Exception:  # noqa: BLE001
        pass
    if board_file.exists():
        try:
            with open(board_file, "r", encoding="utf-8") as f:
                bdata = json.load(f)
            now = time.time()
            active_leases = [l for l in bdata.get("leases", []) if l.get("expires_at", 0) >= now]
            if active_leases:
                flight_lines = ["* 🛫 Active Agent Flight Board (Concurrent Leases):"]
                for l in active_leases:
                    elapsed = round((now - l.get("checked_out_timestamp", now)) / 60.0, 1)
                    flight_lines.append(f"  -> [{l.get('status', 'EXECUTING')}] Agent: `{l.get('agent_id')}` on `{l.get('mission_id')}` (Worktree: `{l.get('worktree')}`, Elapsed: {elapsed}m)")
                parts.append("\n".join(flight_lines))
        except Exception:
            pass

    if top_mission:
        parts.append(
            f"* Priority Mission Target (Highest Token ROI):\n"
            f"  -> [{top_mission.get('id')}] {top_mission.get('title')}\n"
            f"     Priority Score: {top_mission.get('priority_score')} | Status: {top_mission.get('status')} | Surface: {top_mission.get('execution_surface', 'local')}"
        )
    memory_file = project_root / "context" / "memory.md"
    if memory_file.exists():
        lines = [line.strip() for line in memory_file.read_text(encoding="utf-8").splitlines() if line.strip().startswith("- ")]
        if lines:
            parts.append("* Active Context:\n" + "\n".join(f"  {line}" for line in lines[:4]))
    if not parts:
        return "* Sovereign engineering habitat active. APM capabilities nominal."
    return "\n\n".join(parts)


def get_preflight_brief(project_root: Path) -> str:
    ledger_file = project_root / "chronicle" / "ledger.json"
    entry_count = 0
    if ledger_file.exists():
        try:
            with open(ledger_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entry_count = len(data.get("entries", [])) if isinstance(data, dict) else len(data)
        except Exception:
            pass

    return (f"* Sovereign Memory Engine active (Layer 1 fast-boot + Layer 2 progressive disclosure).\n"
            f"* Living Chronicle indexed: {entry_count} entries recorded in `chronicle/ledger.json`.")


def get_relay_section(project_root: Path) -> str:
    """Predecessor Relay section for the briefing (MISSION-025); empty string if the relay module or its data is unavailable."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import relay
        return relay.render_markdown(relay.build_relay(project_root))
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Generate Guardian Boot Digest (Protocol 128)")
    parser.add_argument("--mode", type=str, default="HOLISTIC", help="Wakeup mode")
    parser.add_argument("--write-digest", action="store_true", help="Explicitly persist boot digest to .agent/learning/guardian_boot_digest.md")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format to stdout")
    args = parser.parse_args()

    project_root = get_project_root()
    git_meta = get_git_telemetry(project_root)
    health_status, checks = check_layer1_health(project_root)
    runtime_checks = check_runtime_telemetry()
    top_mission = get_top_mission(project_root)

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
        get_tactical_priorities(project_root, top_mission),
        "",
        "## III. Pre-Flight Brief (Sovereign Memory Engine)",
        get_preflight_brief(project_root),
        "",
        get_relay_section(project_root),
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
    digest_path_str = None

    if args.write_digest:
        learning_dir = project_root / ".agent" / "learning"
        digest_path = learning_dir / "guardian_boot_digest.md"
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(digest_path, 'w', encoding='utf-8') as f:
            f.write(digest_text)
        digest_path_str = str(digest_path.relative_to(project_root))

    total_time_ms = (time.perf_counter() - PROCESS_START_TIME) * 1000

    if args.format == "markdown":
        print(digest_text)
    else:
        output_payload = {
            "status": "success" if health_status == "GREEN" else "warning",
            "health": health_status,
            "git": git_meta,
            "digest_path": digest_path_str,
            "digest_persisted": bool(args.write_digest),
            "top_incomplete_mission": top_mission,
            "checks": checks,
            "runtime_checks": runtime_checks,
            "total_time_ms": total_time_ms
        }
        print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()

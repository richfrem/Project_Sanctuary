#!/usr/bin/env python3
"""manage_coordination.py - Multi-Agent Concurrency Control & Flight Board CLI.
Purpose: Coordinates parallel execution across heterogeneous AI CLI agents (AGY, Claude Code, Codex, MLX) via optimistic file leases, worktree tracking, and flight board status.
Layer: sanctuary-core / operating-sanctuary-guardian (Layer 1 Multi-Agent Concurrency & Leases).
Key Inputs: .agent/coordination/board.json, .agent/coordination/leases.jsonl, LEARNING/research/missions/*.json, index.jsonl.
Key Outputs: Active coordination board, lease tokens, worktree management, and collision prevention alerts.
Usage:
    python3 manage_coordination.py checkout --mission MISSION-026 --agent antigravity --targets "scripts/manage_coordination.py"
    python3 manage_coordination.py checkout --mission MISSION-031 --agent claude-code --create-worktree   # locks come from the manifest's projected_impact
    python3 manage_coordination.py suggest --count 3      # missions that can run in parallel now (disjoint from leases and each other)
    python3 manage_coordination.py matrix                 # pairwise overlap of open missions
    python3 manage_coordination.py verify --mission MISSION-031 --worktree .worktrees/mission-031   # declared impact vs files actually changed
    python3 manage_coordination.py board
    python3 manage_coordination.py heartbeat --mission MISSION-026 --agent antigravity
    python3 manage_coordination.py checkin --mission MISSION-026 --agent antigravity --status COMPLETED --pr 278
Key Functions: get_project_root(), load_board(), save_board(), normalize_handoff(), checkout(), checkin(), suggest(), build_matrix(), verify_impact(), get_active_board(), heartbeat(), clean_stale(), main().
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import impact  # noqa: E402  (projected impact and overlap analysis, MISSION-031)
import usage_hook  # noqa: E402  (measured usage telemetry at checkout/checkin, MISSION-035)


def get_project_root() -> Path:
    """Resolve the git repository root directory."""
    if os.environ.get("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def get_coordination_paths(root: Path) -> tuple[Path, Path]:
    """Return paths to board.json and leases.jsonl, ensuring parent directories exist."""
    coord_dir = root / ".agent" / "coordination"
    coord_dir.mkdir(parents=True, exist_ok=True)
    return coord_dir / "board.json", coord_dir / "leases.jsonl"


def load_board(root: Path) -> Dict[str, Any]:
    """Load active flight board state."""
    board_file, _ = get_coordination_paths(root)
    if board_file.exists():
        try:
            with open(board_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"leases": []}


def save_board(root: Path, board: Dict[str, Any]) -> None:
    """Save active flight board state atomically."""
    board_file, _ = get_coordination_paths(root)
    temp_file = board_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(board, f, indent=2)
    temp_file.replace(board_file)


def append_lease_event(root: Path, event: Dict[str, Any]) -> None:
    """Append lease lifecycle audit event to leases.jsonl."""
    _, log_file = get_coordination_paths(root)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def check_conflicts(active_leases: List[Dict[str, Any]], targets: List[str],
                    resources: List[str], agent: str, mission: str) -> List[Dict[str, Any]]:
    """Check if requested targets or resources conflict with an active lease held by another agent."""
    conflicts = []
    now = time.time()
    for lease in active_leases:
        # Ignore expired leases or leases belonging to the same agent and mission
        if lease.get("expires_at", 0) < now:
            continue
        if lease.get("agent_id") == agent and lease.get("mission_id") == mission:
            continue

        leased_targets = lease.get("locked_targets", [])
        leased_resources = lease.get("exclusive_resources", [])

        # Target overlap check
        matched_targets = [t for t in targets if any(impact.paths_overlap(t, lt) for lt in leased_targets)]

        # Resource overlap check
        matched_resources = [r for r in resources if r in leased_resources]

        if matched_targets or matched_resources:
            conflicts.append({
                "holding_agent": lease.get("agent_id"),
                "holding_mission": lease.get("mission_id"),
                "conflicting_targets": matched_targets,
                "conflicting_resources": matched_resources,
                "worktree": lease.get("worktree"),
                "expires_in_sec": round(lease.get("expires_at", 0) - now, 1)
            })

    return conflicts


def checkout(root: Path, mission: str, agent: str, targets: Optional[List[str]] = None,
             resources: Optional[List[str]] = None, worktree: Optional[str] = None,
             ttl_minutes: int = 60, force: bool = False, create_worktree: bool = False,
             allow_unlocked: bool = False, no_usage: bool = False) -> Dict[str, Any]:
    """Check out a mission, register locks on flight board, and optionally create worktree.

    Locks come from the mission manifest's `projected_impact` (paths and resources) plus any manual --targets/--resources. A mission with
    neither a declaration nor any targets or resources is refused (an unlocked lease can never conflict, which defeats the board) unless allow_unlocked.
    Shared generated files (index.jsonl, ledgers, ...) are never locked; they are regenerated after merging.
    """
    board = load_board(root)
    manual_targets = [t.strip() for t in (targets or []) if t.strip()]
    manual_resources = [r.strip() for r in (resources or []) if r.strip()]
    declared = impact.load_manifest_impact(root, mission)
    if declared is None and not manual_targets and not manual_resources and not allow_unlocked:
        return {"status": "error", "mission": mission,
                "error": (f"{mission} has no projected_impact in LEARNING/research/missions/{mission}.json and no --targets were given; "
                          "refusing an unlocked lease (it could never conflict, so it would not protect anyone). Declare projected_impact, "
                          "pass --targets or --resources, or pass --allow-unlocked.")}
    extra_generated = declared.get("shared_generated", []) if declared else []
    targets = impact.lock_set((declared["paths"] if declared else []) + manual_targets, extra_generated)
    resources = list(dict.fromkeys((declared["resources"] if declared else []) + manual_resources))
    impact_source = ("manifest+manual" if declared and manual_targets else "manifest" if declared else "manual" if manual_targets else "unlocked")

    now = time.time()
    expires_at = now + (ttl_minutes * 60)

    # Clean out expired leases
    active_leases = [l for l in board.get("leases", []) if l.get("expires_at", 0) >= now]

    conflicts = check_conflicts(active_leases, targets, resources, agent, mission)
    if conflicts and not force:
        return {
            "status": "conflict",
            "message": f"Cannot check out {mission}: resource/target conflict detected.",
            "conflicts": conflicts
        }

    # Remove any existing lease for this agent and mission to update it cleanly
    active_leases = [l for l in active_leases if not (l.get("agent_id") == agent and l.get("mission_id") == mission)]

    worktree_path = worktree or f".worktrees/{agent}-{mission.lower()}"
    if create_worktree:
        wt_abs = root / worktree_path
        if not wt_abs.exists():
            branch_name = f"feat/{agent}-{mission.lower()}"
            try:
                subprocess.run(
                    ["git", "worktree", "add", "-b", branch_name, str(wt_abs), "origin/main"],
                    cwd=root, capture_output=True, text=True, check=True
                )
            except Exception as e:
                return {"status": "error", "error": f"Failed to create git worktree: {e}"}

    lease_record = {
        "agent_id": agent,
        "mission_id": mission,
        "status": "EXECUTING",
        "worktree": worktree_path,
        "locked_targets": targets,
        "exclusive_resources": resources,
        "impact_source": impact_source,
        "checked_out_at": datetime.now(timezone.utc).isoformat(),
        "checked_out_timestamp": now,
        "last_heartbeat": now,
        "expires_at": expires_at,
        "ttl_minutes": ttl_minutes
    }
    usage_start = usage_hook.start_snapshot(no_usage)
    if usage_start:
        lease_record["usage_start"] = usage_start
        usage_hook.save_latest(root, usage_start)
    active_leases.append(lease_record)
    board["leases"] = active_leases
    save_board(root, board)

    # Append to immutable event log
    event = dict(lease_record)
    event.pop("usage_start", None)  # keep the append-only log small; the snapshot lives on the lease until checkin
    event["event_type"] = "CHECKOUT"
    append_lease_event(root, event)

    # Mark the mission in progress. Write into the NEW worktree's copy when there is one, so the main tree is not left dirty
    # (a dirty main tree aborts `git pull` and collides with the mission's own PR); fall back to this tree for lease-only checkouts.
    rel = Path("LEARNING") / "research" / "missions" / f"{mission}.json"
    wt_copy = root / worktree_path / rel
    mission_file = wt_copy if wt_copy.exists() else root / rel
    if mission_file.exists():
        try:
            with open(mission_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["status"] = "IN_PROGRESS"
            data["assignee"] = agent
            data["worktree"] = worktree_path
            with open(mission_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    return {
        "status": "success",
        "action": "CHECKOUT",
        "mission": mission,
        "agent": agent,
        "worktree": worktree_path,
        "locked_targets": targets,
        "exclusive_resources": resources,
        "impact_source": impact_source,
        "regenerate_after_merge": list(impact.SHARED_GENERATED),
        "usage_headroom_warnings": usage_hook.headroom_warnings(usage_start, agent),
        "expires_at": datetime.fromtimestamp(expires_at, timezone.utc).isoformat()
    }


MAX_HANDOFF_CHARS = 600


def normalize_handoff(text: Optional[str]) -> Optional[str]:
    """Collapse whitespace and cap a handoff note so one line of the ledger stays readable (None if empty)."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return None
    return cleaned if len(cleaned) <= MAX_HANDOFF_CHARS else cleaned[: MAX_HANDOFF_CHARS - 1].rstrip() + "\u2026"


def checkin(root: Path, mission: str, agent: str, status: str = "COMPLETED",
            pr: Optional[int] = None, clean_worktree: bool = False, handoff: Optional[str] = None,
            task_class: str = "mission", tokens: Optional[int] = None, quality: Optional[int] = None,
            retries: int = 0, no_usage: bool = False, model: Optional[str] = None) -> Dict[str, Any]:
    """Check in a mission, release locks, and optionally tear down git worktree.

    `handoff` is an optional note for the successor (what was done, decisions, what NOT to redo, next step). It is stored on the CHECKIN
    event and surfaced by the boot briefing's Predecessor Relay (see relay.py).
    """
    board = load_board(root)
    now = time.time()
    matching_leases = [l for l in board.get("leases", []) if l.get("agent_id") == agent and l.get("mission_id") == mission]
    remaining_leases = [l for l in board.get("leases", []) if not (l.get("agent_id") == agent and l.get("mission_id") == mission)]

    board["leases"] = remaining_leases
    save_board(root, board)

    event = {
        "event_type": "CHECKIN",
        "agent_id": agent,
        "mission_id": mission,
        "outcome_status": status,
        "pr_number": pr,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    note = normalize_handoff(handoff)
    if note:
        event["handoff"] = note
    usage = usage_hook.finish(root, matching_leases[0] if matching_leases else {}, mission, agent, task_class=task_class,
                              tokens=tokens, quality=quality, retries=retries, no_usage=no_usage, model=model)
    if usage:
        event["usage"] = usage
    append_lease_event(root, event)

    if clean_worktree and matching_leases:
        wt_path = matching_leases[0].get("worktree")
        if wt_path:
            wt_abs = root / wt_path
            if wt_abs.exists():
                subprocess.run(["git", "worktree", "remove", "--force", str(wt_abs)], cwd=root, capture_output=True)

    return {
        "status": "success",
        "action": "CHECKIN",
        "mission": mission,
        "agent": agent,
        "released_locks": matching_leases[0].get("locked_targets", []) if matching_leases else [],
        "pr_number": pr,
        "handoff": note,
        "usage": usage
    }


def _open_candidates(root: Path) -> List[Dict[str, Any]]:
    """Open missions (PROPOSED/STAGED) with their priority and normalised projected impact (None if undeclared)."""
    out = []
    mdir = root / "LEARNING" / "research" / "missions"
    for p in sorted(mdir.glob("MISSION-*.json")) if mdir.exists() else []:
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("status") in impact.OPEN_STATUSES:
            out.append({"id": m["id"], "priority": m.get("priority_score", 0), "title": m.get("title", ""),
                        "impact": impact.normalise_impact(m.get("projected_impact"))})
    return out


def suggest(root: Path, count: int = 3) -> Dict[str, Any]:
    """Up to `count` open, registered missions that are free of active leases and mutually disjoint, by priority."""
    active = get_active_board(root)
    cands = _open_candidates(root)
    res = impact.dispatchable(cands, active, count)
    titles = {c["id"]: c["title"] for c in cands}
    res["selected_titles"] = {i: titles.get(i, "") for i in res["selected"]}
    res["active_leases"] = [l.get("mission_id") for l in active]
    return res


def build_matrix(root: Path) -> Dict[str, Any]:
    """Pairwise overlap of the open missions that declare projected_impact."""
    cands = _open_candidates(root)
    impacts = {c["id"]: c["impact"] for c in cands if c["impact"]}
    return {"registered": sorted(impacts), "conflicts": impact.overlap_matrix(impacts),
            "unregistered": sorted(c["id"] for c in cands if not c["impact"])}


def verify_impact(root: Path, mission: str, worktree: Optional[str] = None, base: str = "origin/main") -> Dict[str, Any]:
    """Compare the declared impact with the files a worktree actually changed (committed against `base`, staged, unstaged, untracked)."""
    declared = impact.load_manifest_impact(root, mission)
    lease = next((l for l in load_board(root).get("leases", []) if l.get("mission_id") == mission), None)
    if declared is None and lease:
        declared = {"paths": list(lease.get("locked_targets", [])), "resources": [], "shared_generated": []}
    if declared is None:
        return {"status": "error", "error": f"{mission} has no projected_impact and no active lease to compare against."}
    wt = root / (worktree or (lease or {}).get("worktree") or ".")
    changed: List[str] = []
    for cmd in (["git", "diff", "--name-only", f"{base}...HEAD"], ["git", "diff", "--name-only", "HEAD"],
                ["git", "ls-files", "--others", "--exclude-standard"]):
        res = subprocess.run(cmd, cwd=wt, capture_output=True, text=True)
        if res.returncode == 0:
            changed += [l for l in res.stdout.splitlines() if l.strip()]
    changed = sorted(set(changed))
    own = {"paths": declared["paths"] + [f"LEARNING/research/missions/{mission}.json"],   # a mission always edits its own manifest
           "shared_generated": declared.get("shared_generated", [])}
    undeclared = impact.undeclared_changes(own, changed)
    return {"status": "success", "mission": mission, "declared_paths": declared["paths"], "changed_files": changed,
            "undeclared": undeclared, "ok": not undeclared}


def heartbeat(root: Path, mission: str, agent: str, extend_minutes: int = 30) -> Dict[str, Any]:
    """Update heartbeat timestamp and extend lease TTL for active mission."""
    board = load_board(root)
    now = time.time()
    updated = False
    new_expires_at = now + (extend_minutes * 60)

    for lease in board.get("leases", []):
        if lease.get("agent_id") == agent and lease.get("mission_id") == mission:
            lease["last_heartbeat"] = now
            lease["expires_at"] = new_expires_at
            updated = True
            break

    if updated:
        save_board(root, board)
        return {
            "status": "success",
            "action": "HEARTBEAT",
            "mission": mission,
            "agent": agent,
            "expires_at": datetime.fromtimestamp(new_expires_at, timezone.utc).isoformat()
        }
    return {
        "status": "error",
        "error": f"No active lease found for agent '{agent}' on mission '{mission}'."
    }


def get_active_board(root: Path) -> List[Dict[str, Any]]:
    """Return active flight board entries with human duration telemetry."""
    board = load_board(root)
    now = time.time()
    active = []

    for lease in board.get("leases", []):
        exp = lease.get("expires_at", 0)
        is_stale = (exp < now)
        elapsed_sec = round(now - lease.get("checked_out_timestamp", now), 1)
        heartbeat_ago = round(now - lease.get("last_heartbeat", now), 1)

        active.append({
            "agent_id": lease.get("agent_id"),
            "mission_id": lease.get("mission_id"),
            "status": "STALE" if is_stale else lease.get("status", "EXECUTING"),
            "worktree": lease.get("worktree"),
            "locked_targets": lease.get("locked_targets", []),
            "exclusive_resources": lease.get("exclusive_resources", []),
            "elapsed_minutes": round(elapsed_sec / 60.0, 1),
            "heartbeat_seconds_ago": heartbeat_ago,
            "is_stale": is_stale
        })

    return active


def render_board_text(active_leases: List[Dict[str, Any]]) -> str:
    """Format flight board entries as terminal-friendly table."""
    if not active_leases:
        return "🛫 Agent Coordination Flight Board: (Empty - No active leases held)"

    lines = ["🛫 Agent Coordination Flight Board (Active Leases):"]
    lines.append("-" * 80)
    for l in active_leases:
        status_tag = f"[{l['status']}]"
        targets = ", ".join(l["locked_targets"]) if l["locked_targets"] else "(None)"
        resources = ", ".join(l["exclusive_resources"]) if l["exclusive_resources"] else "(None)"
        lines.append(f"  {status_tag:10} Agent: {l['agent_id']:12} Mission: {l['mission_id']}")
        lines.append(f"             Worktree: {l['worktree']}")
        lines.append(f"             Locked Targets:   {targets}")
        lines.append(f"             Locked Resources: {resources}")
        lines.append(f"             Elapsed: {l['elapsed_minutes']}m | Heartbeat: {l['heartbeat_seconds_ago']}s ago")
        lines.append("")
    lines.append("-" * 80)
    return "\n".join(lines)


def clean_stale(root: Path) -> Dict[str, Any]:
    """Clean out expired leases from the board."""
    board = load_board(root)
    now = time.time()
    active = []
    removed = []

    for lease in board.get("leases", []):
        if lease.get("expires_at", 0) >= now:
            active.append(lease)
        else:
            removed.append(lease)

    board["leases"] = active
    save_board(root, board)
    return {"status": "success", "removed_count": len(removed), "removed": removed}


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Concurrency Control & Flight Board CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Command to execute")

    # Checkout
    p_co = subparsers.add_parser("checkout", help="Check out a mission and lock targets")
    p_co.add_argument("--mission", required=True, help="Mission identifier (e.g. MISSION-026)")
    p_co.add_argument("--agent", required=True, help="Agent identifier (e.g. antigravity, claude-code)")
    p_co.add_argument("--targets", help="Comma-separated target file/directory paths")
    p_co.add_argument("--resources", help="Comma-separated exclusive resource identifiers")
    p_co.add_argument("--worktree", help="Relative path to isolated worktree")
    p_co.add_argument("--create-worktree", action="store_true", help="Automatically create git worktree")
    p_co.add_argument("--ttl", type=int, default=60, help="Lease time-to-live in minutes")
    p_co.add_argument("--force", action="store_true", help="Force checkout despite target conflicts")
    p_co.add_argument("--no-usage", action="store_true", help="Skip the zero-token usage snapshot stored on the lease")
    p_co.add_argument("--allow-unlocked", action="store_true", help="Allow a lease with no declared impact and no --targets (it protects nobody)")
    p_co.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # Checkin
    p_ci = subparsers.add_parser("checkin", help="Check in a mission and release locks")
    p_ci.add_argument("--mission", required=True, help="Mission identifier")
    p_ci.add_argument("--agent", required=True, help="Agent identifier")
    p_ci.add_argument("--status", default="COMPLETED", choices=["COMPLETED", "CANCELLED", "ABANDONED"])
    p_ci.add_argument("--pr", type=int, help="Merged PR number")
    p_ci.add_argument("--task-class", default="mission", help="Task class for measured-usage records (e.g. mission, drafting, refactor, review)")
    p_ci.add_argument("--tokens", type=int, help="Vendor-reported tokens used (required to record non-claude agents)")
    p_ci.add_argument("--quality", type=int, choices=[0, 1, 2, 3], help="Outcome quality 0-3 (e.g. from score_contribution)")
    p_ci.add_argument("--retries", type=int, default=0, help="Retries or restarts needed")
    p_ci.add_argument("--model", help="Exact model id (enables dollar cost when it is in the cli-agent model catalog)")
    p_ci.add_argument("--no-usage", action="store_true", help="Skip the usage snapshot and measured record")
    p_ci.add_argument("--handoff", help="Note for the successor agent: what you did, decisions, what NOT to redo, next step")
    p_ci.add_argument("--clean-worktree", action="store_true", help="Remove git worktree directory")
    p_ci.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # Impact analysis (MISSION-031)
    p_mx = subparsers.add_parser("matrix", help="Pairwise overlap of open missions that declare projected_impact")
    p_mx.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    p_sg = subparsers.add_parser("suggest", help="Open missions that can be dispatched in parallel now (disjoint from leases and each other)")
    p_sg.add_argument("--count", type=int, default=3, help="How many missions to suggest")
    p_sg.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    p_vf = subparsers.add_parser("verify", help="Compare declared projected_impact with the files a worktree changed")
    p_vf.add_argument("--mission", required=True, help="Mission identifier")
    p_vf.add_argument("--worktree", help="Worktree path relative to the project root (default: the lease's worktree)")
    p_vf.add_argument("--base", default="origin/main", help="Base ref to diff against")
    p_vf.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # Board
    p_bd = subparsers.add_parser("board", help="View active agent flight board")
    p_bd.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # Heartbeat
    p_hb = subparsers.add_parser("heartbeat", help="Send heartbeat to extend lease TTL")
    p_hb.add_argument("--mission", required=True, help="Mission identifier")
    p_hb.add_argument("--agent", required=True, help="Agent identifier")
    p_hb.add_argument("--extend", type=int, default=30, help="Extend TTL in minutes")
    p_hb.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # Clean Stale
    p_cs = subparsers.add_parser("clean-stale", help="Remove expired leases from board")
    p_cs.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    root = get_project_root()

    if args.subcommand == "checkout":
        targets = [t.strip() for t in args.targets.split(",")] if args.targets else []
        resources = [r.strip() for r in args.resources.split(",")] if args.resources else []
        res = checkout(root, args.mission, args.agent, targets=targets, resources=resources,
                       worktree=args.worktree, ttl_minutes=args.ttl, force=args.force,
                       create_worktree=args.create_worktree, allow_unlocked=args.allow_unlocked,
                       no_usage=args.no_usage)
        if args.format == "json":
            print(json.dumps(res, indent=2))
        else:
            if res.get("status") == "conflict":
                print(f"❌ CHECKOUT BLOCKED by conflict:\n{json.dumps(res['conflicts'], indent=2)}")
            elif res.get("status") == "error":
                print(f"❌ CHECKOUT REFUSED: {res.get('error')}")
            else:
                print(f"✅ Checked out [{args.mission}] for agent [{args.agent}]")
                print(f"   Worktree: {res['worktree']}")
                print(f"   Locks:    {', '.join(res['locked_targets']) or '(none)'}  [{res['impact_source']}]")
                print(f"   Expires:  {res['expires_at']}")
                print(f"   After merging, regenerate (do not hand-merge): {', '.join(res['regenerate_after_merge'])}")
        if res.get("status") in ("conflict", "error"):
            sys.exit(1)

    elif args.subcommand == "matrix":
        res = build_matrix(root)
        if args.format == "json":
            print(json.dumps(res, indent=2))
        else:
            print(f"Registered open missions: {len(res['registered'])} | unregistered (no projected_impact): {len(res['unregistered'])}")
            for mid in res["registered"]:
                print(f"  {mid}: conflicts with {', '.join(res['conflicts'][mid]) or '-'}")

    elif args.subcommand == "suggest":
        res = suggest(root, args.count)
        if args.format == "json":
            print(json.dumps(res, indent=2))
        else:
            print(f"Dispatchable in parallel now ({len(res['selected'])}):")
            for mid in res["selected"]:
                print(f"  - {mid}  {res['selected_titles'].get(mid, '')[:80]}")
            for b in res["blocked"][:8]:
                print(f"  x {b['id']} blocked by {b['by']}")
            print(f"Unregistered (declare projected_impact to include): {len(res['unregistered'])}")

    elif args.subcommand == "verify":
        res = verify_impact(root, args.mission, args.worktree, args.base)
        if args.format == "json":
            print(json.dumps(res, indent=2))
        elif res["status"] == "error":
            print(f"❌ {res['error']}")
        elif res["ok"]:
            print(f"✅ [{args.mission}] changed {len(res['changed_files'])} file(s), all inside the declared impact.")
        else:
            print(f"❌ [{args.mission}] changed files OUTSIDE the declared impact:\n  " + "\n  ".join(res["undeclared"]))
        if res["status"] == "error" or not res.get("ok"):
            sys.exit(1)

    elif args.subcommand == "checkin":
        res = checkin(root, args.mission, args.agent, status=args.status, pr=args.pr,
                      clean_worktree=args.clean_worktree, handoff=args.handoff, task_class=args.task_class,
                      tokens=args.tokens, quality=args.quality, retries=args.retries, no_usage=args.no_usage,
                      model=args.model)
        if args.format == "json":
            print(json.dumps(res, indent=2))
        else:
            print(f"✅ Checked in [{args.mission}] for agent [{args.agent}]. Locks released.")

    elif args.subcommand == "board":
        leases = get_active_board(root)
        if args.format == "json":
            print(json.dumps({"active_leases": leases}, indent=2))
        else:
            print(render_board_text(leases))

    elif args.subcommand == "heartbeat":
        res = heartbeat(root, args.mission, args.agent, extend_minutes=args.extend)
        if args.format == "json":
            print(json.dumps(res, indent=2))
        else:
            print(f"💓 Heartbeat received for [{args.mission}] ({args.agent}). Expires: {res.get('expires_at')}")

    elif args.subcommand == "clean-stale":
        res = clean_stale(root)
        if args.format == "json":
            print(json.dumps(res, indent=2))
        else:
            print(f"🧹 Cleaned {res['removed_count']} stale leases.")


if __name__ == "__main__":
    main()

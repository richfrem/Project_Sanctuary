#!/usr/bin/env python3
"""
manage_backlog.py - Agent-Native Cognitive Backlog & Investment Tracker
======================================================================

Purpose:
    Sovereign, agent-first backlog and falsifiable mission tracker for
    Project Sanctuary. Provides ultra-compact Tier 1 machine indices,
    zero-friction append-only inbox capture, structured mission manifests,
    and automatic deterministic compilation to human Markdown views.

Usage:
    python manage_backlog.py list [--status STATUS] [--format text|json|jsonl]
    python manage_backlog.py get <MISSION-ID>
    python manage_backlog.py capture --idea "..." [--author AGENT] [--tags ...]
    python manage_backlog.py compile
    python manage_backlog.py validate
    python manage_backlog.py record-result --id <MISSION-ID> --verdict CORROBORATED|FALSIFIED|ABANDONED --evidence "..." [--citation "..."] [--lessons "..."]
"""

import sys
import json
import uuid
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "LEARNING").exists():
            return parent
    return Path.cwd()


def get_research_dir(root: Optional[Path] = None) -> Path:
    base = root or get_project_root()
    return base / "LEARNING" / "research"


def calculate_priority_score(leverage: int, depth: int, cost: float, friction: int) -> float:
    """Calculates priority score: (leverage * depth) / (cost * friction)."""
    if cost <= 0:
        cost = 0.05
    if friction <= 0:
        friction = 1
    return round((leverage * depth) / (cost * friction), 2)


class BacklogManager:
    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or get_project_root()
        self.research_dir = self.root_dir / "LEARNING" / "research"
        self.missions_dir = self.research_dir / "missions"
        self.inbox_file = self.research_dir / "inbox.jsonl"
        self.index_file = self.research_dir / "index.jsonl"
        self.markdown_file = self.research_dir / "backlog.md"
        self.events_file = self.root_dir / ".agent" / "learning" / "events.jsonl"

        self.missions_dir.mkdir(parents=True, exist_ok=True)
        self.research_dir.mkdir(parents=True, exist_ok=True)

    def load_missions(self) -> List[Dict[str, Any]]:
        """Loads all mission JSON manifests from missions_dir."""
        missions: List[Dict[str, Any]] = []
        if not self.missions_dir.exists():
            return missions

        for f in sorted(self.missions_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    missions.append(data)
            except Exception as e:
                print(f"[WARN] Failed to parse {f.name}: {e}", file=sys.stderr)
        return missions

    def get_mission(self, mission_id: str) -> Optional[Dict[str, Any]]:
        target = self.missions_dir / f"{mission_id}.json"
        if target.exists():
            with open(target, "r", encoding="utf-8") as fp:
                return json.load(fp)
        return None

    def save_mission(self, mission: Dict[str, Any]) -> Path:
        mission_id = mission["id"]
        target = self.missions_dir / f"{mission_id}.json"
        with open(target, "w", encoding="utf-8") as fp:
            json.dump(mission, fp, indent=2)
        return target

    def capture_inbox(self, idea: str, author: str = "agent", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Appends a raw unvetted idea to inbox.jsonl."""
        now = datetime.now(timezone.utc)
        seed_id = f"SEED-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
        entry = {
            "id": seed_id,
            "created_at": now.isoformat(),
            "author": author,
            "idea": idea.strip(),
            "tags": tags or [],
            "status": "CAPTURED"
        }
        with open(self.inbox_file, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry) + "\n")
        return entry

    def record_result(self, mission_id: str, verdict: str, evidence: str,
                      citation: str = "", lessons: Optional[List[str]] = None,
                      agent: str = "agent") -> Dict[str, Any]:
        """Records an empirical trial verdict, evidence, and lessons learned."""
        mission = self.get_mission(mission_id)
        if not mission:
            raise ValueError(f"Mission {mission_id} not found.")

        valid_verdicts = {"CORROBORATED", "FALSIFIED", "ABANDONED"}
        if verdict.upper() not in valid_verdicts:
            raise ValueError(f"Invalid verdict '{verdict}'. Must be one of {valid_verdicts}")

        now = datetime.now(timezone.utc)
        mission["verdict"] = verdict.upper()
        mission["status"] = "RESOLVED"
        mission["updated_at"] = now.isoformat()

        if "evidence" not in mission:
            mission["evidence"] = []

        evidence_entry = {
            "timestamp": now.isoformat(),
            "agent": agent,
            "citation": citation,
            "details": evidence
        }
        mission["evidence"].append(evidence_entry)

        if lessons:
            if "lessons_learned" not in mission:
                mission["lessons_learned"] = []
            mission["lessons_learned"].extend(lessons)

        self.save_mission(mission)

        # Log event into .agent/learning/events.jsonl if root has it
        if self.events_file.parent.exists():
            event = {
                "event_type": "MISSION_RESOLVED",
                "mission_id": mission_id,
                "timestamp": now.isoformat(),
                "verdict": verdict.upper(),
                "citation": citation,
                "agent": agent,
                "lessons": lessons or []
            }
            with open(self.events_file, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(event) + "\n")

        # Recompile index and markdown
        self.compile()
        return mission

    def compile(self) -> None:
        """Validates all missions, writes Tier 1 index.jsonl, and generates backlog.md."""
        missions = self.load_missions()

        # Recalculate priority scores and validate
        for m in missions:
            scores = m.get("scores", {})
            leverage = scores.get("strategic_leverage", m.get("strategic_leverage", 1))
            depth = scores.get("epistemic_depth", m.get("epistemic_depth", 1))
            cost = scores.get("cost_weight", m.get("cost_weight", 1.0))
            friction = scores.get("operational_friction", m.get("operational_friction", 1))

            calc_score = calculate_priority_score(leverage, depth, cost, friction)
            scores["strategic_leverage"] = leverage
            scores["epistemic_depth"] = depth
            scores["cost_weight"] = cost
            scores["operational_friction"] = friction
            scores["priority_score"] = calc_score
            m["scores"] = scores
            m["priority_score"] = calc_score
            self.save_mission(m)

        # Sort by priority score descending
        missions.sort(
            key=lambda x: (
                x.get("priority_score", 0.0),
                x.get("scores", {}).get("strategic_leverage", 0),
                x.get("scores", {}).get("epistemic_depth", 0)
            ),
            reverse=True
        )

        # 1. Write Tier 1 index.jsonl
        with open(self.index_file, "w", encoding="utf-8") as fp:
            for rank, m in enumerate(missions, start=1):
                index_item = {
                    "rank": rank,
                    "id": m["id"],
                    "title": m["title"],
                    "status": m["status"],
                    "priority_score": m["priority_score"],
                    "execution_surface": m.get("execution_surface", "local"),
                    "verdict": m.get("verdict"),
                    "path": f"LEARNING/research/missions/{m['id']}.json"
                }
                fp.write(json.dumps(index_item) + "\n")

        # 2. Write Human-Readable backlog.md
        md_lines = [
            "<!-- THIS FILE IS AUTOMATICALLY GENERATED BY manage_backlog.py compile. DO NOT EDIT BY HAND -->",
            "# Prioritized Sovereign Research Backlog",
            "",
            "## 1. Executive Summary & Philosophy",
            "",
            "Project Sanctuary operates as a **sovereign engineering habitat, anti-Asch epistemic platform, and autonomous agent substrate**.",
            "Autonomous agents maintain their backlogged inquiries natively using **Tier 1 machine index (`index.jsonl`)** and structured JSON manifests (`missions/*.json`).",
            "This document is a generated human projection representing the current portfolio.",
            "",
            "---",
            "",
            "## 2. Priority & ROI Formula",
            "",
            "$$\\text{Priority Score} = \\frac{\\text{Strategic Leverage (1–5)} \\times \\text{Epistemic Depth (1–5)}}{\\text{Cost Weight (0.05–4.5)} \\times \\text{Operational Friction (1–3)}}$$",
            "",
            "---",
            "",
            "## 3. Prioritized Missions Portfolio",
            "",
            "| Rank | ID | Title | Leverage | Depth | Cost | Friction | Score | Status | Verdict |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for rank, m in enumerate(missions, start=1):
            s = m.get("scores", {})
            verdict_str = f"`{m['verdict']}`" if m.get("verdict") else "-"
            status_str = f"`{m.get('status', 'PROPOSED')}`"
            md_lines.append(
                f"| **#{rank}** | `{m['id']}` | **{m['title']}** | {s.get('strategic_leverage', 1)} | "
                f"{s.get('epistemic_depth', 1)} | {s.get('cost_weight', 1.0)}x | {s.get('operational_friction', 1)} | "
                f"**{m.get('priority_score', 0.0)}** | {status_str} | {verdict_str} |"
            )

        md_lines.extend(["", "---", "", "## 4. Mission Deep-Dives", ""])

        for rank, m in enumerate(missions, start=1):
            md_lines.append(f"### {m['id']}: {m['title']} (Rank #{rank})")
            md_lines.append(f"- **Status**: `{m.get('status', 'PROPOSED')}`")
            if m.get("verdict"):
                md_lines.append(f"- **Empirical Verdict**: `{m['verdict']}`")
            md_lines.append(f"- **Execution Surface**: `{m.get('execution_surface', 'local')}`")
            md_lines.append(f"- **Target Platform**: {m.get('target_platform', 'Sanctuary core')}")
            md_lines.append(f"- **Hypothesis (Popperian Prediction)**:\n  > {m.get('hypothesis', 'N/A')}")
            md_lines.append(f"- **Falsification Gate**:\n  > {m.get('falsification_criteria', 'N/A')}")

            evidence = m.get("evidence", [])
            if evidence:
                md_lines.append("- **Empirical Evidence & Trials**:")
                for ev in evidence:
                    cit = f" ({ev.get('citation')})" if ev.get("citation") else ""
                    md_lines.append(f"  * `{ev.get('timestamp', '')}` [{ev.get('agent', 'agent')}]{cit}: {ev.get('details', '')}")

            lessons = m.get("lessons_learned", [])
            if lessons:
                md_lines.append("- **Lessons Learned & Epistemic Compounding**:")
                for les in lessons:
                    md_lines.append(f"  * {les}")

            md_lines.append("")

        with open(self.markdown_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(md_lines) + "\n")

    def validate(self) -> List[str]:
        """Validates all missions for schema correctness and score consistency."""
        errors = []
        missions = self.load_missions()

        for m in missions:
            mid = m.get("id", "UNKNOWN")
            if not m.get("title"):
                errors.append(f"[{mid}] Missing title")
            if not m.get("hypothesis"):
                errors.append(f"[{mid}] Missing hypothesis (Popperian prediction required)")
            if not m.get("falsification_criteria"):
                errors.append(f"[{mid}] Missing falsification criteria")

            scores = m.get("scores", {})
            leverage = scores.get("strategic_leverage", m.get("strategic_leverage"))
            depth = scores.get("epistemic_depth", m.get("epistemic_depth"))
            cost = scores.get("cost_weight", m.get("cost_weight"))
            friction = scores.get("operational_friction", m.get("operational_friction"))

            if None in (leverage, depth, cost, friction):
                errors.append(f"[{mid}] Missing one or more scoring dimensions")
            else:
                expected = calculate_priority_score(leverage, depth, cost, friction)
                actual = m.get("priority_score")
                if actual is not None and abs(actual - expected) > 0.05:
                    errors.append(f"[{mid}] Stored priority score ({actual}) != calculated ({expected})")

        return errors


def main():
    parser = argparse.ArgumentParser(description="Agent-Native Cognitive Backlog Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_p = subparsers.add_parser("list", help="List missions")
    list_p.add_argument("--status", help="Filter by status")
    list_p.add_argument("--format", choices=["text", "json", "jsonl"], default="text", help="Output format")

    # get
    get_p = subparsers.add_parser("get", help="Get a single mission manifest")
    get_p.add_argument("id", help="Mission ID (e.g. MISSION-001)")

    # capture
    cap_p = subparsers.add_parser("capture", help="Capture a raw idea to inbox.jsonl")
    cap_p.add_argument("--idea", required=True, help="Idea text")
    cap_p.add_argument("--author", default="agent", help="Author/agent identifier")
    cap_p.add_argument("--tags", nargs="*", help="Tags")

    # compile
    subparsers.add_parser("compile", help="Compile manifests into index.jsonl and backlog.md")

    # validate
    subparsers.add_parser("validate", help="Validate all mission manifests")

    # record-result
    res_p = subparsers.add_parser("record-result", help="Record empirical verdict and evidence")
    res_p.add_argument("--id", required=True, help="Mission ID")
    res_p.add_argument("--verdict", required=True, choices=["CORROBORATED", "FALSIFIED", "ABANDONED"])
    res_p.add_argument("--evidence", required=True, help="Evidence details")
    res_p.add_argument("--citation", default="", help="Citation ref, commit SHA, or file link")
    res_p.add_argument("--lessons", nargs="*", help="Key lessons learned")
    res_p.add_argument("--agent", default="agent", help="Executing agent identity")

    args = parser.parse_args()
    mgr = BacklogManager()

    if args.command == "list":
        missions = mgr.load_missions()
        if args.status:
            missions = [m for m in missions if m.get("status", "").upper() == args.status.upper()]

        # Sort by priority score descending
        missions.sort(key=lambda x: x.get("priority_score", 0.0), reverse=True)

        if args.format == "json":
            print(json.dumps(missions, indent=2))
        elif args.format == "jsonl":
            for rank, m in enumerate(missions, start=1):
                item = {
                    "rank": rank,
                    "id": m["id"],
                    "title": m["title"],
                    "status": m.get("status"),
                    "priority_score": m.get("priority_score"),
                    "verdict": m.get("verdict"),
                    "execution_surface": m.get("execution_surface")
                }
                print(json.dumps(item))
        else:
            print(f"{'Rank':<5} {'ID':<12} {'Status':<10} {'Score':<8} {'Surface':<14} {'Title'}")
            print("-" * 80)
            for rank, m in enumerate(missions, start=1):
                print(f"#{rank:<4} {m['id']:<12} {m.get('status',''):<10} {m.get('priority_score',0.0):<8.1f} {m.get('execution_surface','local'):<14} {m.get('title','')}")

    elif args.command == "get":
        mission = mgr.get_mission(args.id)
        if not mission:
            print(f"Mission '{args.id}' not found.", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(mission, indent=2))

    elif args.command == "capture":
        entry = mgr.capture_inbox(args.idea, author=args.author, tags=args.tags)
        print(f"Captured inbox item: {entry['id']}")

    elif args.command == "compile":
        mgr.compile()
        print("Compiled missions into index.jsonl and generated backlog.md successfully.")

    elif args.command == "validate":
        errors = mgr.validate()
        if errors:
            print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)
        else:
            print("Validation passed. All mission manifests conform to schema and invariants.")

    elif args.command == "record-result":
        updated = mgr.record_result(
            args.id,
            args.verdict,
            args.evidence,
            citation=args.citation,
            lessons=args.lessons,
            agent=args.agent
        )
        print(f"Recorded verdict '{updated['verdict']}' for {args.id}. Backlog recompiled.")


if __name__ == "__main__":
    main()

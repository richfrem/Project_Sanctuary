---
description: Protocol for Dual-Loop Agentic Learning (Outer Loop Strategy + Inner Loop Execution)
tier: 1
track: B
---

# Dual-Loop Agent Architecture (Protocol 133)

**Objective:** High-Velocity Code Execution with Strategic Oversight.
**Diagram:** `docs/architecture_diagrams/workflows/dual_loop_architecture.mmd`
**Skill:** `.agent/skills/dual-loop-supervisor/SKILL.md` (authoritative reference for agents)
**Constraint:** **Token Efficiency** for the Inner Loop (Opus) is paramount.

---

## 1. Concept: The Two Loops

| Component | Agent | Role | Focus | Git Authority |
|-----------|-------|------|-------|---------------|
| **Outer Loop** | Antigravity (Gemini) | **Strategy & Oversight** | "What & Why" | **Repo Owner** (Branch/Merge) |
| **Inner Loop** | Claude Code (Opus) | **Tactical Execution** | "How" | **None** (No Git Cmds) |

**Key Invariant:** The Inner Loop receives a Strategy Packet, executes code, and signals completion. It does NOT run git commands, Learning Loop phases, or exploratory changes.

---

## 2. The Workflow (5 Phases)

### Phase I: Strategy (Outer Loop)

> **Prerequisite**: Complete Protocol 128 (Learning Loop) Phases I-IV (Scout -> Audit) before entering this execution branch.

1. **Specify**: `/spec-kitty.specify` → `spec.md`
2. **Plan**: `/spec-kitty.plan` → `plan.md`
3. **Task**: `/spec-kitty.tasks` → `tasks.md` + `tasks/WP-*.md`
4. **Verify**: `python3 tools/orchestrator/verify_workflow_state.py --feature <SLUG> --phase tasks`
5. **Workspace**: `/spec-kitty.implement <WP-ID>` → creates worktree
6. **Distill**: Generate Strategy Packet (minimal context for Inner Loop)
   ```bash
   python3 tools/orchestrator/dual_loop/generate_strategy_packet.py \
     --tasks-file kitty-specs/<FEATURE>/tasks.md --task-id <WP-ID>
   ```
   Output: `.agent/handoffs/task_packet_NNN.md`

### Phase II: Hand-off (Human Bridge)
1. Outer Loop signals: "Ready for Execution"
2. User switches terminal to Claude Code
3. User runs: `claude "Read .agent/handoffs/task_packet_NNN.md. Execute the mission. Do NOT use git."`

### Phase III: Execution (Inner Loop)
1. Opus reads packet, writes code, runs tests
2. **Scope**: RESTRICTED to the Strategy Packet
3. **Completion**: Opus reports "Done" when acceptance criteria are met

### Phase IV: Verification (Outer Loop)
1. User returns to Antigravity
2. Run verification:
   ```bash
   python3 tools/orchestrator/dual_loop/verify_inner_loop_result.py \
     --packet .agent/handoffs/task_packet_NNN.md --verbose
   ```
3. Run state check:
   ```bash
   python3 tools/orchestrator/verify_workflow_state.py --wp <WP-ID> --phase review
   ```
4. **Pass**: Commit in worktree, update task lane to `done`
5. **Fail**: Auto-generates `correction_packet_NNN.md` → repeat Phase II

### Phase V: Retrospective (Protocol 128 Phase VIII)
1. **Outer → Inner**: "Did the code meet the spec?" (Quality)
2. **Inner → Outer**: "Was the Strategy Packet clear?" (Clarity — user proxies)
3. **Refinement**: Update packet template if clarity score is low

---

## 3. Fallback: Branch-Direct Mode

If the worktree is empty or inaccessible to the Inner Loop:
1. Inner Loop implements directly on the **feature branch**
2. Outer Loop notes this in the **friction log** (`.agent/frictionlogs/`)
3. Outer Loop reviews **branch diff** instead of worktree diff

This is a degraded mode — investigate and fix for next iteration.

---

## 4. Token Efficiency Protocol

1. **No Chat History**: Inner Loop starts fresh for each Task Packet
2. **File Focus**: Packet specifies exactly which files are relevant
3. **Zero-Shot Preference**: Perfect spec → one-pass execution
4. **Context Injection**: Packet generator auto-includes spec/plan excerpts (truncated to 2000 chars)

---

## 5. Protocol 128 Integration

| P128 Phase | Dual-Loop Role | Notes |
|------------|---------------|-------|
| I (Scout) | Outer Loop boots, orients | Reads boot files + spec context |
| II-III (Synthesis/Gate) | Outer Loop plans, user approves | Strategy Packet generated |
| IV (Audit) | Outer Loop snapshots before delegation | Pre-execution checkpoint |
| *Execution* | **Inner Loop** codes | No git, no P128 phases |
| *Verification* | Outer Loop inspects output | `verify_inner_loop_result.py` |
| V (RLM Synthesis) | Outer Loop (Automated) | Cognitive Hologram generation |
| VI-IX (Seal→End) | Outer Loop closure | Standard seal/persist/retro/end |

---

## 6. Tooling Quick Reference

| Tool | Purpose | Path |
|------|---------|------|
| Strategy Packet Generator | Distill task → minimal packet | `tools/orchestrator/dual_loop/generate_strategy_packet.py` |
| Inner Loop Verifier | Check output vs acceptance criteria | `tools/orchestrator/dual_loop/verify_inner_loop_result.py` |
| Workflow Runner | End-to-end orchestrator | `tools/orchestrator/dual_loop/run_workflow.py` |
| Workflow State Checker | Artifact/worktree integrity | `tools/orchestrator/verify_workflow_state.py` |
| Task Lane CLI | WP status management | `.kittify/scripts/tasks/tasks_cli.py` |

---

## 7. Cross-References

- **Skill (agents read this)**: `.agent/skills/dual-loop-supervisor/SKILL.md`
- **Learning Loop**: `.agent/workflows/sanctuary_protocols/sanctuary-learning-loop.md` (Protocol 128)
- **Spec Kitty**: `.agent/skills/spec_kitty_workflow/SKILL.md`
- **Diagram**: `docs/architecture_diagrams/workflows/dual_loop_architecture.mmd`
- **Spec**: `kitty-specs/001-dual-loop-agent-architecture/spec.md`

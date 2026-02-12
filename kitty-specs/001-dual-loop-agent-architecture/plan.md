# Implementation Plan: Dual-Loop Agent Architecture
*Path: kitty-specs/001-dual-loop-agent-architecture/plan.md*


**Branch**: `001-dual-loop-agent-architecture` | **Date**: 2026-02-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/kitty-specs/001-dual-loop-agent-architecture/spec.md`

**Note**: This template is filled in by the `/spec-kitty.plan` command. See `src/specify_cli/missions/software-dev/command-templates/plan.md` for the execution workflow.

The planner will not begin until all planning questions have been answered—capture those answers in this document before progressing to later phases.

## Summary

The Dual-Loop Agent Architecture introduces a hierarchical execution model separating strategic planning (Outer Loop) from tactical implementation (Inner Loop). This system leverages a high-agency "Mission-Based" hand-off protocol where the Strategic Controller (Antigravity/Gemini) defines high-level specs and tasks, while the Tactical Executor (Opus/Claude Code) performs autonomous coding and testing loops. This plan delivers a **new workflow, skill, and toolset** to make this process seamless and reusable. Crucially, it focuses on **Token Efficiency** for the Inner Loop (Opus 4.6), ensuring it receives only the minimal, high-signal "Strategy Packet" required to execute, rather than full conversation history.

## Technical Context

**Language/Version**: Markdown (for protocols/specs), Mermaid (for diagrams), Python 3.11+ (for skills/tools)
**Primary Dependencies**: `cortex-mcp` (existing), `spec-kitty` (existing context)
**Storage**: File-based (Markdown artifacts, JSON logs)
**Testing**: Manual verification of workflow execution (Protocol Testing)
**Target Platform**: Local Dev Environment (Mac/Linux)
**Project Type**: Process & Documentation (with supporting CLI tools)
**Performance Goals**: N/A (Process optimization)
**Constraints**: Must align with Protocol 128 (Learning Loop) and Spec-Driven Development.
**Scale/Scope**: Core architecture change affecting all future "learning" workflows.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Human Gate**: All state-changing operations (Outer to Inner hand-off) must be explicit user actions? **YES**, user manually triggers Opus execution.
- **Spec-Driven**: Does this follow Spec->Plan->Tasks? **YES**, we are currently in that flow.
- **Protocol 128**: Does this integrate with the Learning Loop? **YES**, it is designed to be the engine for Phase II-IV.

## Project Structure

### Documentation (this feature)

```
kitty-specs/001-dual-loop-agent-architecture/
├── plan.md              # This file
├── research.md          # N/A (Architecture definitions)
├── data-model.md        # N/A (Process flow)
├── quickstart.md        # "How to run a Dual-Loop Session" guide
├── contracts/           # Interaction schemas (Outer -> Inner protocol)
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```
.agent/
├── workflows/
│   └── sanctuary_protocols/
│       └── dual-loop-learning.md      # The new protocol definition
├── skills/
│   └── dual-loop-supervisor/          # New skill for Outer Loop
│       └── SKILL.md
└── templates/                         # Templates for hand-off artifacts

docs/
└── architecture_diagrams/
    └── workflows/
        └── dual_loop_architecture.mmd # Visualization
```

**Structure Decision**: We are extending the `.agent/` configuration space rather than building a new application source tree. This is an infrastructure upgrade.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| New Skill | To formalize the Supervisor role | Ad-hoc prompting is inconsistent and violates Protocol 128 rigor |
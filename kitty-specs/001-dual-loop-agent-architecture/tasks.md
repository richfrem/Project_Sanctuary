# Dual-Loop Agent Architecture - Tasks

**Feature**: `001-dual-loop-agent-architecture`
**Workflow**: `/spec-kitty.tasks`

## Phase 1: Architecture & Documentation (Outer Loop)
<!--
  Dependencies: None
  Focus: Define the protocols and structures.
-->

- [x] **Define the Supervisor Skill (`dual-loop-supervisor`)** <!-- id: 1 -->
  - Create `.agent/skills/dual-loop-supervisor/SKILL.md`.
  - Define the prompt structure for generating "Strategy Packets" (the clean task definition for Opus).
  - Define the verification logic (how to check Opus's work).
  - **Constraint**: Must reference Protocol 128 (Learning Loop) as the parent protocol.

- [ ] **Create the Workflow Shortcut (`/sanctuary-dual-loop`)** <!-- id: 2 -->
  - Create `.agent/workflows/sanctuary_protocols/dual-loop-learning.md`.
  - Ensure it includes the "Hand-off" steps clearly for the human operator.
  - Link to `docs/architecture_diagrams/workflows/dual_loop_architecture.mmd`.

- [ ] **Develop the "Strategy Packet" Template** <!-- id: 3 -->
  - Create `.agent/templates/handoff/strategy-packet-template.md`.
  - Sections:
    - **Context**: Minimal sufficient context (filenames, lines).
    - **Objective**: Use "Mission-Type" orders (e.g., "Refactor X to achieve Y").
    - **Constraints**: Safety/Style rules.
    - **Acceptance Criteria**: Verifiable output checks.
  - **Goal**: Token efficiency for Opus.

## Phase 2: Tooling & Automation (Inner Loop Enablement)
<!--
  Dependencies: Phase 1
  Focus: Build the tools to make the hand-off smooth.
-->

- [ ] **Implement `generate_strategy_packet.py`** <!-- id: 4 -->
  - Create a script in `tools/orchestrator/dual_loop/`.
  - Input: Current `tasks.md` selection.
  - Output: A standalone `.md` file ready to be pasted/fed to Claude Code.
  - Function: Distills the task + context into the template.

- [ ] **Implement `verify_inner_loop_result.py`** <!-- id: 5 -->
  - Create a script in `tools/orchestrator/dual_loop/`.
  - Function: Diff check on the repository.
  - Function: Run relevant tests.
  - Output: A "Verification Report" for the Outer Loop to read.

- [ ] **Register Tools in Inventory** <!-- id: 6 -->
  - Update `tools/tool_inventory.json` with the new scripts.
  - Ensure they are accessible to Antigravity (Outer Loop).

## Phase 3: Pilot Run (Verification)
<!--
  Dependencies: Phase 2
  Focus: Prove it works.
-->

- [ ] **Pilot Experiment: "Refactor a small utility"** <!-- id: 7 -->
  - Use the new workflow to plan a small refactor (e.g., in `tools/utils`).
  - Generate the Strategy Packet.
  - **Human Action**: Hand off to Opus.
  - **Human Action**: Hand back to Antigravity.
  - Verify the result and the "smoothness" of the process.

- [ ] **Retrospective & Polish** <!-- id: 8 -->
  - Update the workflow based on friction points found in the pilot.
  - Finalize `quickstart.md` for the Dual-Loop system.

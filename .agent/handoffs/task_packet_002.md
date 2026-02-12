# Mission: Implement Dual-Loop Tooling (Packet #002)
**(Strategy Packet for Inner Loop / Opus)**

> **Objective:** Build the Python CLI tools that automate the "Strategy Packet Generation" and "Verification" steps, using the prompts defined in the Supervisor Skill.

## 1. Context
- **Spec**: `kitty-specs/001-dual-loop-agent-architecture/spec.md`
- **Plan**: `kitty-specs/001-dual-loop-agent-architecture/plan.md`
- **Skill**: `.agent/skills/dual-loop-supervisor/SKILL.md`
- **Goal**: We need executable scripts (`tools/orchestrator/dual_loop/*.py`) that the Outer Loop (Antigravity) can call to generate packets and verify results.

## 2. Tasks
Create or update the following files:

### A. `tools/orchestrator/dual_loop/generate_strategy_packet.py`
- **Function**: CLI tool that takes a `tasks.md` path and a task ID.
- **Logic**:
    1.  Reads the task definition.
    2.  Loads `prompts/strategy_generation.md`.
    3.  (Simulation) Prints the *packet content* to stdout (or saves to `.agent/handoffs/`).
- **Constraint**: Use standard `argparse`. No external LLM calls needed yet (just the skeleton/logic).

### B. `tools/orchestrator/dual_loop/verify_inner_loop_result.py`
- **Function**: CLI tool that takes a `strategy_packet_path`.
- **Logic**:
    1.  Runs `git diff --stat` (simulated or real).
    2.  Loads `prompts/verification.md`.
    3.  (Simulation) Prints the *verification report* structure to stdout.

### C. Update `SKILL.md` (Documentation Enhancement)
- **Task**: Replace the existing ASCII art diagram with a Mermaid code block.
- **Source**: Use the logic from `docs/architecture_diagrams/workflows/dual_loop_architecture.mmd` as inspiration, but keep it simple for the Skill doc.

## 3. Constraints
- **NO GIT COMMANDS**: Just write the Python files and update the Markdown.
- **Dependencies**: Use standard library where possible.
- **Structure**: Ensure `tools/orchestrator/dual_loop/` has an `__init__.py`.

## 4. Acceptance Criteria
- [ ] `generate_strategy_packet.py` exists and runs with `--help`.
- [ ] `verify_inner_loop_result.py` exists and runs with `--help`.
- [ ] `SKILL.md` contains a valid `mermaid` block instead of ASCII.
- [ ] `__init__.py` exists in the new directory.

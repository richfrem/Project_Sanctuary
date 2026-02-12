# Mission: Implement Dual-Loop Supervisor Skill
**(Strategy Packet for Inner Loop / Opus)**

> **Objective:** Create the "Brain" of the Outer Loop - the Skill that defines how Antigravity manages the Dual-Loop process.

## 1. Context
- **Spec**: `kitty-specs/001-dual-loop-agent-architecture/spec.md`
- **Plan**: `kitty-specs/001-dual-loop-agent-architecture/plan.md`
- **Goal**: We need a formal Skill definition (`SKILL.md`) that documents the prompts and logic for:
    1.  Generating Strategy Packets (Distilling tasks).
    2.  Verifying Inner Loop Output (Quality checks).

## 2. Tasks
Create the following files in `.agent/skills/dual-loop-supervisor/`:

### A. The Skill Definition (`SKILL.md`)
- **Format**: Standard Skill Markdown.
- **Description**: "Orchestration logic for Dual-Loop Agent Architecture (Protocol 133)."
- **Commands**:
    - `generate_packet`: Uses `prompts/strategy_generation.md`.
    - `verify_output`: Uses `prompts/verification.md`.

### B. The Prompts (`prompts/`)
- `strategy_generation.md`: A system prompt for Antigravity to take a `tasks.md` item and turn it into a minimal, token-efficient instruction set for Opus.
- `verification.md`: A system prompt for Antigravity to review a git diff and decide Pass/Fail.

## 3. Constraints
- **NO GIT COMMANDS**: Just write the files. The Outer Loop handles version control.
- **Token Efficiency**: The prompts you write should encourage brevity.
- **Protocol 128**: Reference the Learning Loop phases (I-X) in the SKILL.md.

## 4. Acceptance Criteria
- `SKILL.md` exists and follows the template.
- `prompts/strategy_generation.md` exists.
- `prompts/verification.md` exists.

# Feature Specification: Dual Loop Standalone Kit (WP-006)

## 1. Goal
Create a portable, self-contained **"Dual Loop Kit"** that bundles all Protocol 128 (Learning Loop) and Protocol 133 (Dual Loop) assets into a single distributable artifact. This allows other repositories to easily adopt the Sanctuary workflow by unpacking a single file.

## 2. Context
*   **Problem**: The Dual Loop and Learning Loop systems are powerful but scattered across `tools/`, `.agent/skills`, `.agent/workflows`, and `docs/`. Sharing them requires manual copying or complex git submodules.
*   **Solution**: Use the `context-bundler` pattern to create a "Kit" – a directory with a manifest that lists all dependencies. This kit can be "compiled" into a single `dual-loop-kit.md` for distribution.

## 3. Requirements

### 3.1. Directory Structure
Create `tools/standalone/dual-loop-kit/` with:
*   `dual-loop-manifest.json`: The inventory of files.
*   `README.md`: Overview of the kit.
*   `INSTALL.md`: How to install/unpack.
*   `UNPACK_INSTRUCTIONS.md`: Protocol for hydration (Standard Bundler Protocol).
*   `prompt.md`: Identity/Instructions for the Kit itself (if it acts as an agent).

### 3.2. Content Inventory (The Manifest)
The kit MUST include:
*   **Diagrams**:
    *   `docs/architecture_diagrams/workflows/dual_loop_architecture.mmd`
    *   `docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd`
*   **Skills**:
    *   `.agent/skills/dual-loop-supervisor/SKILL.md`
    *   `.agent/skills/learning-loop/SKILL.md`
    *   `.agent/skills/spec_kitty_workflow/SKILL.md` (Dependency)
*   **Workflows (Protocols)**:
    *   `.agent/workflows/sanctuary_protocols/dual-loop-learning.md`
    *   `.agent/workflows/sanctuary_protocols/sanctuary-learning-loop.md`
    *   `.agent/workflows/sanctuary_protocols/sanctuary-start.md`
    *   `.agent/workflows/sanctuary_protocols/sanctuary-end.md`
*   **Orchestrator Scripts**:
    *   `tools/orchestrator/dual_loop/*.py`
    *   `tools/orchestrator/workflow_manager.py`
    *   `plugins/spec-kitty-plugin/skills/spec-kitty-agent/scripts/verify_workflow_state.py`
*   **Templates**:
    *   `.agent/templates/workflow/*.md`

### 3.3. Capabilities
*   **Self-Contained**: The kit should include a `bundle.py` (or reference the standard one) to verify it can pack itself.
*   **Reusable Skill**: Define a new skill (or update `dual-loop-supervisor`) to include instructions on how to use/maintain this kit.

## 4. User Experience
1.  **User**: "I want to install Dual Loop in my new repo."
2.  **Action**: User copies `dual-loop-kit.md` to new repo.
3.  **Action**: User (or Agent) reads `dual-loop-kit.md` and runs the "Unpack" protocol.
4.  **Result**: New repo has `tools/orchestrator`, `.agent/skills`, and `docs/` populated with the Dual Loop system.

## 5. Constraints
*   **No Code Changes to Core Logic**: This task is about *bundling* existing code, not refactoring it.
*   **Paths**: Must handle relative path resolution correctly so it works when unpacked in a different root.

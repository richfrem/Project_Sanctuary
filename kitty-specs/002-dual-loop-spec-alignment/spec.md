# Spec: Dual-Loop / Spec Kitty Integration (Protocol 133 v2)

**Status**: Draft
**Owner**: Antigravity (Outer Loop)
**Feature ID**: 002-dual-loop-spec-alignment

## 1. Problem Statement
The current Dual-Loop Architecture (Protocol 133) requires significant manual orchestration. The user must explicitly run `generate_strategy_packet.py`, construct the launch command for Opus, and then manually trigger verification. Furthermore, the task status in `tasks.md` (the Kanban board) is not automatically updated when work is verified, leading to drift between the codebase state and the project plan. The current `tasks.md` parsing logic is also regex-based and brittle.

## 2. Goals
1.  **Seamless Integration**: Embed the "Strategy Packet" generation directly into the standard Spec Kitty workflow (`/spec-kitty.implement`).
2.  **Robust Parsing**: Replace regex-based task parsing with a structured parser that understands the Spec Kitty `tasks.md` format natively.
3.  **Automated Kanban**: Ensure that successful verification (`verify_inner_loop_result.py`) automatically updates the corresponding task in `tasks.md` to "Done" (`[x]`).
4.  **Work Package Alignment**: Ensure that Spec Kitty Work Packages (WP-NN) map 1:1 to Dual-Loop Strategy Packets.

## 3. User Stories
- **As the Outer Loop Agent**, I want to run `/spec-kitty.implement WP-01` and have it automatically generate the corresponding Strategy Packet for the Inner Loop.
- **As the Inner Loop Agent**, I want Strategy Packets to have a standard, consistent format that references the specific `tasks.md` line items I need to execute.
- **As the Project Manager**, I want the `tasks.md` file to reflect the real-time status of work, updated automatically when code is verified.

## 4. Implementation Plan (High Level)
1.  **Refactor `generate_strategy_packet.py`**:
    - Support structued parsing of `tasks.md`.
    - Accept `WP-NN` identifiers.
2.  **Enhance `verify_inner_loop_result.py`**:
    - Add logic to update `tasks.md` (checkbox toggling) upon successful verification.
3.  **Integrate with Workflow**:
    - Update `plugins/agent-loops/personas/README.md` with the new integrated commands.
    - (Optionally) Create a wrapper script `spec-kitty-dual-loop.sh` that chains `implement` -> `generate` -> `launch`.

## 5. Constraints
- **Token Efficiency**: The generated Strategy Packet must defined *scoped context* only (no full repo dump).
- **No Git in Inner Loop**: The Inner Loop agent remains restricted from git commands.
- **Standard Library**: Tools should remain dependency-light (standard python libraries).

## 6. Acceptance Criteria
- [ ] `generate_strategy_packet.py` correctly parses standard `tasks.md` format without fragile regex.
- [ ] Running the verification tool with `--update-status` marks the task as complete in `tasks.md`.
- [ ] A documented workflow exists for "One-Command Handoff" (Spec -> Packet -> Launch).

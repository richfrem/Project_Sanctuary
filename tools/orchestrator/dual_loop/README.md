# Dual-Loop Orchestration Tools

**Protocol**: 133 (Dual-Loop Agent Architecture)
**Purpose**: CLI tools for the Outer Loop (Antigravity) to generate and verify work for the Inner Loop (Opus).

## Quick Start (Manual Integration with Spec Kitty)

Until native integration lands in `/spec-kitty.implement`, use this workflow:

1.  **Define Tasks**:
    ```bash
    /spec-kitty.tasks  # Generates kitty-specs/MMM/tasks.md
    ```

2.  **Generate Packet**:
    ```bash
    python3 tools/orchestrator/dual_loop/generate_strategy_packet.py \
      --tasks-file kitty-specs/001/tasks.md \
      --task-id 1 \
      --output .agent/handoffs/manual_packet_001.md
    ```

3.  **Execute (Inner Loop)**:
    ```bash
    claude "Read .agent/handoffs/manual_packet_001.md — Execute the mission. NO GIT."
    ```

4.  **Verify (Outer Loop)**:
    ```bash
    python3 tools/orchestrator/dual_loop/verify_inner_loop_result.py \
      --packet .agent/handoffs/manual_packet_001.md
    ```

## Roadmap
- [ ] Automate step 2 and 4 inside `/spec-kitty.implement` and `/spec-kitty.review`.

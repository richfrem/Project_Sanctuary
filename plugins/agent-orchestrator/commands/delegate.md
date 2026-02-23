# /agent-orchestrator_delegate — Handoff to Inner Loop

**Purpose:** This command prepares the technical context required for an **Inner Loop (Executor)** to implement a specific Work Package (WP). It bridges the strategic plan to tactical execution.

---

## 🏗️ The Step-by-Step Handoff

1. **Create Isolated Workspace (MANDATORY)**:
   ```bash
   spec-kitty implement WP-NN
   ```
   *Action:* Creates a worktree and checks out a feature branch.

2. **Generate Strategy Packet**:
   ```bash
   /agent-orchestrator_delegate --wp WP-NN
   ```
   *Action:* Leverages `agent_handoff.py` to bundle the spec, plan, and target files into a single mission document.
   *Output:* `.agent/handoffs/task_packet_NN.md`

3. **Packet Review Checklist**:
   Confirm the generated packet contains:
   - [ ] **Objective**: The specific goal of this WP.
   - [ ] **Acceptance Criteria**: Verifiable "Definition of Done".
   - [ ] **File Scope**: Exactly which files the Inner Loop is allowed to touch.
   - [ ] **Strategic Constraints**: ADRs or architectural rules to follow.
   - [ ] **Anti-Simulation Reminder**: Reminds the executor to provide command output proof.

4.  **Launch Inner Loop**:
    Instruct the user or launch tool:
    > "Strategy packet ready at `.agent/handoffs/task_packet_NN.md`. Launching Inner Loop executor."

---

## ⛔ The Git Wall (Inner Loop Constraint)

> [!CAUTION]
> **The Inner Loop executor MUST NOT run Git commands.** 
> All state changes, commits, and merges are the exclusive responsibility of the Outer Loop (Orchestrator).

---

## 🛤️ Context Awareness
- **Track A (Factory):** Packets may be auto-generated from standardized WP templates.
- **Track B (Discovery):** Packets require manual refinement of instructions to handle ambiguity.

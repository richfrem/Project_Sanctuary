---
description: Protocol for Dual-Loop Agentic Learning (Outer Loop Strategy + Inner Loop Execution)
tier: 1
track: B
---

# Dual-Loop Agent Architecture (Protocol 133)

**Objective:** High-Velocity Code Execution with Strategic Oversight.
**Diagram:** `docs/architecture_diagrams/workflows/dual_loop_architecture.mmd`
**Constraint:** **Token Efficiency** for the Inner Loop (Opus) is paramount.

---

## 1. Concept: The Two Loops

| Component | Agent | Role | Focus | Token Window | Git Authority |
|-----------|-------|------|-------|--------------|---------------|
| **Outer Loop** | Antigravity (Gemini) | **Strategy & Oversight** | "What & Why" | High (Full Context) | **Repo Owner** (Branch/Merge) |
| **Inner Loop** | Claude Code (Opus) | **Tactical Execution** | "How" | **Low (Task-Specific)** | **Worktree Only** (No Git Cmds) |

## 2. The Workflow

### Phase I: Strategy (Outer Loop)
1.  **Initiate**: User runs `/sanctuary-dual-loop "Goal Description"`.
2.  **Spec Generation**:
    *   System checks for existing spec.
    *   **If New**: System internally triggers `/spec-kitty.specify` to define the architecture/task.
    *   **If Existing**: System loads the current context.
3.  **Workspace Prep**: Antigravity runs `/spec-kitty.implement <WP-ID>`.
    *   Creates isolated worktree: `.worktrees/feature-WP01`.
    *   Isolates Opus from main repo noise.
4.  **Distill**: Creates a **Minimal Context Object** for Opus *inside* the worktree.
    *   *CRITICAL:* Do NOT send the whole repo context. Send only:
        *   The specific file(s) to edit.
        *   The specific constraints.
        *   The acceptance criteria.
    *   Artifact: `.agent/handoffs/task_packet_NNN.md`

### Phase II: Hand-off (Human Bridge)
1.  **Trigger**: Antigravity signals: "Ready for Execution."
2.  **Switch**: User switches terminal to `claude`.
3.  **Execute**: User runs: `claude "Execute the task defined in .agent/handoffs/task_packet_NNN.md"`

### Phase III: Execution (Inner Loop)
1.  **Code**: Opus writes code, runs tests, fixes bugs.
2.  **Constraint**: Opus is RESTRICTED to the scope of the Packet.
3.  **Completion**: Opus reports "Done" when tests pass.

### Phase IV: Verification (Outer Loop)
1.  **Switch**: User returns to Antigravity.
2.  **Verify**: Antigravity inspects the *diff* (not the chat history).
3.  **Judge**:
    *   **Pass**: Run `/sanctuary-seal`.
    *   **Fail**: Generate `correction_packet_NNN.md` and repeat Phase II.

### Phase V: Dual-Loop Retrospective (Protocol 128 Phase VIII)
1.  **Bidirectional Feedback**:
    *   **Outer -> Inner**: "Did the code meet the spec?" (Quality Check)
    *   **Inner -> Outer**: "Was the Strategy Packet clear?" (Clarity Check - User proxies this feedback)
2.  **Refinement**: If the Packet was unclear, Antigravity updates the `strategy-packet-template.md` for next time.
3.  **Recursion**: This feedback loop improves the *next* cycle's efficiency.

---

## 3. Token Efficiency Protocol

To ensure Opus 4.6 (expensive) is used efficiently:
1.  **No Chat History**: The Inner Loop starts fresh for each Task Packet.
2.  **File Focus**: The Task Packet must specify exactly which files are relevant.
3.  **Zero-Shot Preference**: Aim for Opus to solve it in one go (or few turns) based on a perfect spec, rather than long conversational debugging.

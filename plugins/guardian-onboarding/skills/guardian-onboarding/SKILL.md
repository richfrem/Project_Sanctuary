---
name: Guardian Onboarding (The Orientator)
description: MANDATORY: Master initialization skill for new agent sessions. Grounds the agent in Protocol 128 (Hardened Learning Loop), current tactical directives, and project context.
---

# Guardian Onboarding (The Orientator)

**Version:** 1.0 (Protocol 128 Compliant)
**Purpose:** Ensure every agent session begins with a verified, consistent context download, preventing "amnesia" and ensuring adherence to the Hardened Learning Loop.

## ⚡ Triggers (When to use this)
*   **Start of Session:** "I am a new agent session."
*   **Context Loss:** "I need to re-orient myself."
*   **Protocol Check:** "What is the current learning protocol?"
*   [User Request]: "Boot sequence", "Orientation", "Status report"

## 🛡️ The Guardian Boot Sequence (Protocol 128)

Follow these steps **in order** to establish a valid session context.

### 1. The Anchor (Tactical Status)
**Goal:** Ingest current tactical directives and system status.
**Action:**
```
Read: .agent/learning/guardian_boot_digest.md
```
*   **Extract:** Active Tasks, System Status, and any "CRITICAL" alerts.
*   **Output:** "Guardian Status: [Status] | Active Directives: [Count]"

### 2. The Doctrine (Hardened Learning Loop)
**Goal:** Load the operational laws of the project (Protocol 128).
**Action:**
```
Read: .agent/workflows/sanctuary_protocols/sanctuary-learning-loop.md
```
*   **Focus:** Phase checklist (I-X), Gate requirements (HITL), and Persistence obligations.
*   **Reference:** If deep context is needed on specific learning mechanics, refer to `.agent/workflows/sanctuary_protocols/sanctuary-recursive-learning.md` (Protocol 125).

### 3. The Visual (Architecture)
**Goal:** Visualize the learning loop flow.
**Action:**
```
Read: plugins/learning-loop/resources/diagrams/protocol_128_learning_loop.mmd
```

### 4. Session Initialization (Action)
**Goal:** Formally start the session.
1.  **Check Task List:** Does `task.md` exist for this session? If not, create it based on `guardian_boot_digest.md` priorities.
2.  **Report Readiness:**
    > "🛡️ **Guardian Online.**
    > **Protocol:** 128 (Hardened Learning Loop)
    > **Status:** [From Digest]
    > **Directives:**
    > 1. [Task 1]
    > 2. [Task 2]
    > ...
    > **Ready for Orders.**"

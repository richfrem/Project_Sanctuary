---
description: "Start the outer loop: specify, plan, and generate tasks via spec-kitty"
---

# /agent-orchestrator_plan — Outer Loop: Strategy Entry Point

**Purpose:** This is the **UNIVERSAL STARTING POINT** for all features, refactors, and analysis tasks. It initiates the **Outer Loop Strategy Phase**, ensuring that WHAT we are building and HOW we are building it is defined before a single line of code is written.

---

> **AGENT EXECUTION DIRECTIVE**
> This workflow is **MANDATORY** before any implementation work begins — including deterministic SOPs like `/codify-form`.
> You MUST invoke this workflow via the Skill tool when a user triggers any codify, investigate, or feature task.
> Skipping the orchestrator and proceeding directly to analysis or documentation = **workflow simulation** per Constitution §IV.
> The SDD artifacts (spec.md, plan.md, tasks.md) are non-optional process outputs, not bureaucratic overhead.

---

## 🔄 SDD Lifecycle Orchestration

When called, this command orchestrates the three foundational phases of Spec-Driven Development:

1.  **Phase 0a: Specify** (`/spec-kitty.specify`)
    - Defines the "What" and "Why".
    - Creates `spec.md`.
2.  **Phase 0b: Plan** (`/spec-kitty.plan`)
    - Defines the "How" (Design, Files, Verification).
    - Creates `plan.md`.
3.  **Phase 0c: Tasks** (`/spec-kitty.tasks`)
    - Breaks the plan into actionable **Work Packages (WPs)**.
    - Creates `tasks.md` and `tasks/WP-*.md` prompt files.

---

## 🛤️ Track Awareness (The Context Gate)

The Orchestrator adapts its behavior based on the complexity of the task:

### Track A: Factory (Deterministic)
*   **Context:** Standardized SOPs like `codify-form` or `codify-db-package`.
*   **Behavior:** Initiates with a pre-defined scope. May combine specify/plan/tasks into a single pass if the requirements are deterministic.
*   **Goal:** Consistency and Speed.

### Track B: Discovery (Creative)
*   **Context:** New features, complex refactors, or ambiguous bug fixes.
*   **Behavior:** Requires interactive deep-dives (Specify -> Clarify -> Plan -> Review).
*   **Goal:** Precision and Quality.

---

## 🛠️ Usage

```bash
/agent-orchestrator_plan "[Task Description or Title]"
```

**What happens next:**
1. The agent checks for an existing spec in `kitty-specs/`.
2. If building from an SOP (Factory), it loads the relevant workflow template.
3. It guides the user through the verification of each artifact.
4. **STOP**: You cannot proceed to `/agent-orchestrator_delegate` until `tasks.md` is generated and verified.

---

## 🛡️ Anti-Simulation Policy
The Orchestrator **MUST NOT** manually create `spec.md`, `plan.md`, or `tasks.md`. It must use the `spec-kitty-cli` tools to generate these artifacts to ensure registry integrity.

> [!IMPORTANT]
> Always run `verify_workflow_state.py` after each phase to ensure the environment is ready for implementation.

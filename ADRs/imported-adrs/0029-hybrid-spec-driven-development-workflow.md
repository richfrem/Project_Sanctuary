# ADR-0029: Hybrid Spec-Driven Development Workflow

## Status
Accepted

## Context
The project has established a strong "Factory" of deterministic workflows (`/codify-*`) for documenting legacy artifacts. These are highly efficient for repeatable tasks (e.g., documenting 50 forms).

However, as we move into complex modernization and new feature development, these rigid workflows are insufficient. We need a way to handle **high-ambiguity** tasks that require:
1.  **Specification**: Defining "What" before "How".
2.  **Planning**: Defining architecture before implementation.
3.  **Dynamic Execution**: Adapting the plan based on discovery.

We introduced "Spec-Kit" to handle this, but it created a parallel ecosystem (`spec.md`, `plan.md`) that was disconnected from our existing governance and tools.

## Decision
We will adopt a **Dual-Track Hybrid Workflow** architecture that explicitly categorizes all work into two tracks:

### Track A: Factory (Deterministic)
*   **Purpose**: High-volume, repeatable, low-ambiguity tasks.
*   **Tooling**: `/codify-*`, `/investigate-*`, `/curate-*`.
*   **Protocol**: Strict "Standard Operating Procedure" (SOP). The user says "Go", the Agent follows a script.
*   **Output**: Standardized Documentation, Reports, enriched data.

### Track B: Discovery (Spec-Driven)
*   **Purpose**: High-ambiguity, creative, exploratory tasks (New Features, Architecture Changes).
*   **Tooling**: `/speckit-specify`, `/speckit-plan`, `/speckit-tasks`, `/speckit-implement`.
*   **Protocol**: 4-Step Cycle (Spec -> Plan -> Tasks -> Implement).
*   **Relationship**: Key point: **Track B can spawn Track A tasks.** A Feature Plan might include a task "Run /codify-form on Form X".

### Universal Wrappers & Shims
Both tracks must adhere to the core governance lifecycle, enforced by **Shell Shims**:
1.  **Entry Point**: `scripts/bash/workflow-start.sh` (Constitutional Gate).
2.  **Execution**: `scripts/bash/codify-*.sh` (Orchestrator Shims).
3.  **Exit Point**: `/workflow-end` (Quality Gate).
4.  **Reflection**: `/workflow-retrospective` (Continuous Improvement).

## Consequences
### Positive
*   **Clarity**: Users and Agents know exactly which tool to use based on the problem type (Repetitive vs Creative).
*   **Reuse**: We leverage the powerful "Investigation Modules" (Tier 2) in both tracks.
*   **Safety**: Ambiguous tasks are forced through the Spec/Plan gate, preventing hallucinated architectures.

### Negative
*   **Complexity**: There are now two "modes" of operation.
*   **Maintenance**: We must maintain two sets of templates (SOP templates vs Spec templates).

## References
*   [Agent Workflow Orchestration Design](../architecture/Agent_Workflow_Orchestration_Design.md)
*   [Workflow Composition](../architecture/workflow-composition.md)
*   [Workflow Inventory Analysis](../antigravity/workflow/workflow_inventory_and_analysis.md)

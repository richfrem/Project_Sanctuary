# Spec-Driven Development (SDD) Policy

**Effective Date**: 2026-01-29
**Related Constitution Articles**: IV (Documentation First), V (Test-First), VI (Simplicity)

## 1. Overview
This policy defines the standard workflows for managing work in the Antigravity system. It follows the **Dual-Track** architecture defined in the Constitution.

## 2. The Spec-First Standard
**All significant work** (Features, Modernization, Documentation) must follow the Spec -> Plan -> Task lifecycle.

### 2.1 Track A: Standardized Specs (Factory)
For deterministic, repetitive workflows (e.g., `/codify-rlm-distill`, `/codify-vector-ingest`).
*   **Workflow**: The User invokes a command -> The Agent **Auto-Generates** a Pre-defined Spec/Plan/Task bundle -> The Agent Executes.
*   **Benefit**: Consistency, traceability, and "Human Gate" review even for standard ops.
*   **Artifacts**: Lives in `specs/`.

### 2.2 Track B: Custom Specs (Discovery)
For ambiguous, creative work (e.g., "Design new Auth System").
*   **Workflow**: The User invokes `/spec-kitty.specify` -> The Agent **Drafts** a custom Spec -> User Approves -> Plan -> Execute.
*   **Artifacts**: Lives in `specs/`.

### 2.3 Track C: Micro-Tasks (Maintenance)
For trivial, atomic fixes (e.g., "Fix typo", "Restart server").
*   **Workflow**: Direct execution or simple ticket in `tasks/`.
*   **Constraint**: NO ARCHITECTURAL DECISIONS ALLOWED in Track C.

## 3. The Artifacts
For Tracks A and B, the following artifacts are mandatory in `specs/NNN/`:

### 3.1 The Specification (`spec.md`)
**Template**: `[.agent/templates/workflow/spec-template.md](../../.agent/templates/workflow/spec-template.md)`
*   **Purpose**: Define the "What" and "Why".
*   **Track A**: Populated from Standard Template.
*   **Track B**: Populated from User Interview.

### 3.2 The Implementation Plan (`plan.md`)
**Template**: `[.agent/templates/workflow/plan-template.md](../../.agent/templates/workflow/plan-template.md)`
*   **Purpose**: Define the "How".
*   **Track A**: Standard steps (e.g., "Run miner", "Gen docs").
*   **Track B**: Custom architecture logic.

### 3.3 The Execution Tasks (`tasks.md`)
**Template**: `[.agent/templates/workflow/tasks-template.md](../../.agent/templates/workflow/tasks-template.md)`
*   **Purpose**: Checklist for execution.

## 4. The Workflow Cycle
1.  **Initialize**: User creates spec bundle via `/spec-kitty.specify` (or manual).
2.  **Specify**: Agent creates `spec.md`. User reviews.
3.  **Plan**: Agent creates `plan.md`. Agent self-checks Gates. User reviews.
4.  **Execute**: Agent generates `tasks.md`.
5.  **Implement**: Agent executes tasks using `/spec-kitty.implement`.

## 4. Reverse-Engineering (Migration Context)
When migrating or improving an existing component:
1.  **Discovery**: Run Investigation tools (`/investigate-*`).
2.  **Reverse-Spec**: Use investigation results to populate `spec.md` (Documenting existing behavior).
3.  **Plan**: Create `plan.md` for the migration or improvement.

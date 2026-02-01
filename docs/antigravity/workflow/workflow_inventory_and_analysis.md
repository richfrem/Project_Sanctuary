# Workflow Inventory & Analysis

**Objective**: Audit and classify all Antigravity workflows into the **Dual-Track** architecture (Factory vs. Discovery).

**Created**: 2026-01-30
**Status**: DRAFT (Baseline for Feature 002)

## The Dual-Track Taxonomy

*   **Track A (Factory)**: Deterministic, repeatable, high-volume. The Agent follows a strict script. "SOPs".
*   **Track B (Discovery)**: High-ambiguity, creative, exploratory. The Agent uses `spec.md` and `plan.md` to define its own path. "Spec-Driven".
*   **Shared (Meta-Ops)**: Operational workflows that support both tracks (e.g., git ops, retrospectives).

## Inventory & Classification

| Workflow Name | Current Classification | Classification Rationale | Note |
| :--- | :--- | :--- | :--- |
| **Spec-Kit Core** | | | |
| `/speckit-checklist` | **Track B (Discovery)** | Core SDD loop tool. | |
| `/speckit-clarify` | **Track B (Discovery)** | Clarifies ambiguity in specs. | |
| `/speckit-constitution` | **Track B (Discovery)** | Defines high-level governance. | |
| `/speckit-implement` | **Track B (Discovery)** | Executes the user-defined plan. | |
| `/speckit-plan` | **Track B (Discovery)** | Creates the architecture. | |
| `/speckit-specify` | **Track B (Discovery)** | Creates the requirement. | |
| `/speckit-tasks` | **Track B (Discovery)** | Breaks down the plan. | |
| `/speckit-tasks-to-issues` | **Track B (Discovery)** | PM integration. | |
| **Documentation Factory** | | | |
| `/codify-app` | **Track A (Factory)** | Standardized overview doc. | |
| `/codify-db-*` | **Track A (Factory)** | 10+ DB specific workflows. Strict template. | |
| `/codify-form` | **Track A (Factory)** | The "Form Factory". Strict rigid steps. | |
| `/codify-library` | **Track A (Factory)** | Library documentation standard. | |
| `/codify-menu` | **Track A (Factory)** | Menu documentation standard. | |
| `/codify-report` | **Track A (Factory)** | Report documentation standard. | |
| `/codify-task` | **Shared (Meta-Ops)** | Task creation is universal. | Used by both tracks. |
| **Analysis Modules** | | | |
| `/investigate-*` | **Track A (Factory)** | Reusable modules (Forms, DB, etc). | Called by Track A or B. |
| `/retrieve-*` | **Shared (Meta-Ops)** | Lookups (Vector, RLM, Source). | Universal capability. |
| **Maintenance** | | | |
| `/curate-*` | **Track A (Factory)** | Data hygiene (Inventory, Enriched Links). | Deterministic cleanup. |
| **Operations** | | | |
| `/workflow-start` | **Shared (Meta-Ops)** | Pre-flight check. | Mandatory for all. |
| `/workflow-end` | **Shared (Meta-Ops)** | Post-flight check (PR/Push). | Mandatory for all. |
| `/workflow-retrospective` | **Shared (Meta-Ops)** | Self-improvement loop. | Mandatory for all. |
| **Legacy/Others** | | | |
| `/modernize-form` | **Track A (Factory)** | Proto-typing React code. | Currently deterministic. |
| `/codify-adr` | **Track B (Discovery)** | While structured, ADRs require "Thinking". | Could be Track B? |

## Analysis Findings

1.  **Clear Separation**: The `codify-*` vs `speckit-*` namespace collision is naturally resolved by this taxonomy. `codify` is for **documenting what exists** (Factory). `speckit` is for **building what's new** (Discovery).
2.  **The "Bridge"**: The `investigate-*` workflows are crucial. They are "Factory" modules, but `speckit-plan` (Track B) should fundamentally rely on them to gather context.
3.  **Ambiguity**: `/codify-adr` is currently templated (Factory-like), but the *content* is pure architectural thought (Discovery-like). Recommendation: Keep as Track A for the *file creation*, but the *content generation* is a Track B activity.

## Recommendations for implementation

1.  Update `workflow_inventory.json` with a new field `"track": "factory" | "discovery" | "shared"`.
2.  Update `constitution.md` to explicit rules:
    *   "When using Track A workflows, follow the script exactly."
    *   "When using Track B workflows, create a Spec and Plan first."

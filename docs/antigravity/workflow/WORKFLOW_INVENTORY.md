# Antigravity Workflow Inventory

> **Generated:** 2026-01-31 23:26
> **Total Workflows:** 24


## Track: Discovery

| Command | Tier | Description | Called By |
| :--- | :--- | :--- | :--- |
| `/speckit-analyze` | - | Perform a non-destructive cross-artifact consistency and quality analysis across spec.md, plan.md, and tasks.md after task generation. | - |
| `/speckit-checklist` | - | Generate a custom checklist for the current feature based on user requirements. | - |
| `/speckit-clarify` | - | Identify underspecified areas in the current feature spec by asking up to 5 highly targeted clarification questions and encoding answers back into the spec. | - |
| `/speckit-constitution` | - | Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync. | - |
| `/speckit-implement` | - | Execute the implementation plan by processing and executing all tasks defined in tasks.md | - |
| `/speckit-plan` | - | Execute the implementation planning workflow using the plan template to generate design artifacts. | - |
| `/speckit-specify` | - | Create or update the feature specification from a natural language feature description. | - |
| `/speckit-tasks-to-issues` | - | Convert existing tasks into actionable, dependency-ordered GitHub issues for the feature based on available design artifacts. | - |
| `/speckit-tasks` | - | Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts. | - |

## Track: Factory

| Command | Tier | Description | Called By |
| :--- | :--- | :--- | :--- |
| `/codify-adr` | - | Creates a new Architecture Decision Record (ADR) with proper numbering and template. | - |
| `/post-move-link-check` | - | Run link checker after moving or renaming files/folders | - |
| `/recursive_learning` | - | "Standard operating procedure for Protocol 128 Hardened Learning Loop (Scout -> Synthesize -> Audit -> Seal -> Persist)." | - |
| `/workflow-adr` | - | Manage Architecture Decision Records (ADR) | - |
| `/workflow-audit` | - | Protocol 128 Phase IV - Red Team Audit (Capture Snapshot) | - |
| `/workflow-chronicle` | - | Manage Chronicle Entries (Journaling) | - |
| `/workflow-end` | 1 | Standard post-flight closure for all codify/investigate workflows. Handles human review, git commit, PR verification, and cleanup. | - |
| `/workflow-ingest` | - | Run RAG Ingestion (Protocol 128 Phase IX) | - |
| `/workflow-learning-loop` | - | Orchestrates a Protocol 128 Learning Loop within the standard Spec-First lifecycle. | - |
| `/workflow-persist` | - | Protocol 128 Phase VI - Soul Persistence (Broadcast to Hugging Face) | - |
| `/workflow-retrospective` | 1 | Mandatory self-retrospective and continuous improvement check after completing any codify workflow. | /workflow-end |
| `/workflow-scout` | - | Protocol 128 Phase I - The Learning Scout (Debrief & Orientation) | - |
| `/workflow-seal` | - | Protocol 128 Phase V - The Technical Seal (Snapshot & Validation) | - |
| `/workflow-start` | 1 | Universal pre-flight and Spec initialization for all workflows. Determines work type and ensures Spec-Plan-Tasks exist. | - |
| `/workflow-task` | - | Manage Maintenance Tasks (Kanban) | - |

## Quick Reference (All)

| Command | Track | Description |
| :--- | :--- | :--- |
| `/codify-adr` | Factory | Creates a new Architecture Decision Record (ADR) with proper numbering and template. |
| `/post-move-link-check` | Factory | Run link checker after moving or renaming files/folders |
| `/recursive_learning` | Factory | "Standard operating procedure for Protocol 128 Hardened Learning Loop (Scout -> Synthesize -> Audit -> Seal -> Persist)." |
| `/speckit-analyze` | Discovery | Perform a non-destructive cross-artifact consistency and quality analysis across spec.md, plan.md, and tasks.md after task generation. |
| `/speckit-checklist` | Discovery | Generate a custom checklist for the current feature based on user requirements. |
| `/speckit-clarify` | Discovery | Identify underspecified areas in the current feature spec by asking up to 5 highly targeted clarification questions and encoding answers back into the spec. |
| `/speckit-constitution` | Discovery | Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync. |
| `/speckit-implement` | Discovery | Execute the implementation plan by processing and executing all tasks defined in tasks.md |
| `/speckit-plan` | Discovery | Execute the implementation planning workflow using the plan template to generate design artifacts. |
| `/speckit-specify` | Discovery | Create or update the feature specification from a natural language feature description. |
| `/speckit-tasks` | Discovery | Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts. |
| `/speckit-tasks-to-issues` | Discovery | Convert existing tasks into actionable, dependency-ordered GitHub issues for the feature based on available design artifacts. |
| `/workflow-adr` | Factory | Manage Architecture Decision Records (ADR) |
| `/workflow-audit` | Factory | Protocol 128 Phase IV - Red Team Audit (Capture Snapshot) |
| `/workflow-chronicle` | Factory | Manage Chronicle Entries (Journaling) |
| `/workflow-end` | Factory | Standard post-flight closure for all codify/investigate workflows. Handles human review, git commit, PR verification, and cleanup. |
| `/workflow-ingest` | Factory | Run RAG Ingestion (Protocol 128 Phase IX) |
| `/workflow-learning-loop` | Factory | Orchestrates a Protocol 128 Learning Loop within the standard Spec-First lifecycle. |
| `/workflow-persist` | Factory | Protocol 128 Phase VI - Soul Persistence (Broadcast to Hugging Face) |
| `/workflow-retrospective` | Factory | Mandatory self-retrospective and continuous improvement check after completing any codify workflow. |
| `/workflow-scout` | Factory | Protocol 128 Phase I - The Learning Scout (Debrief & Orientation) |
| `/workflow-seal` | Factory | Protocol 128 Phase V - The Technical Seal (Snapshot & Validation) |
| `/workflow-start` | Factory | Universal pre-flight and Spec initialization for all workflows. Determines work type and ensures Spec-Plan-Tasks exist. |
| `/workflow-task` | Factory | Manage Maintenance Tasks (Kanban) |
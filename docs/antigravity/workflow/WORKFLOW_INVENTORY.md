# Antigravity Workflow Inventory

> **Generated:** 2026-02-01 16:27
> **Total Workflows:** 26


## Track: Discovery

| Command | Tier | Description | Called By |
| :--- | :--- | :--- | :--- |
| `/spec-kitty.analyze` | - | Perform a non-destructive cross-artifact consistency and quality analysis across spec.md, plan.md, and tasks.md after task generation. | - |
| `/spec-kitty.checklist` | - | Generate a custom checklist for the current feature based on user requirements. | - |
| `/spec-kitty.clarify` | - | Identify underspecified areas in the current feature spec by asking up to 5 highly targeted clarification questions and encoding answers back into the spec. | - |
| `/spec-kitty.constitution` | - | Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync. | - |
| `/spec-kitty.implement` | - | Execute the implementation plan by processing and executing all tasks defined in tasks.md | - |
| `/spec-kitty.plan` | - | Execute the implementation planning workflow using the plan template to generate design artifacts. | - |
| `/spec-kitty.specify` | - | Create or update the feature specification from a natural language feature description. | - |
| `/spec-kitty.tasks-to-issues` | - | Convert existing tasks into actionable, dependency-ordered GitHub issues for the feature based on available design artifacts. | - |
| `/spec-kitty.tasks` | - | Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts. | - |

## Track: Factory

| Command | Tier | Description | Called By |
| :--- | :--- | :--- | :--- |
| `/adr-manage` | - | Creates a new Architecture Decision Record (ADR) with proper numbering and template. | - |
| `/post-move-link-check` | - | Run link checker after moving or renaming files/folders | - |
| `/adr-manage` | - | Manage Architecture Decision Records (ADR) | - |
| `/sanctuary-audit` | - | Protocol 128 Phase IV - Red Team Audit (Capture Snapshot) | - |
| `/bundle-manage` | - | Create a markdown bundle from a set of files using a manifest. | - |
| `/sanctuary-chronicle` | - | Manage Chronicle Entries (Journaling) | - |
| `/sanctuary-end` | 1 | Standard post-flight closure for all codify/investigate workflows. Handles human review, git commit, PR verification, and cleanup. | - |
| `/sanctuary-ingest` | - | Run RAG Ingestion (Protocol 128 Phase IX) | - |
| `/sanctuary-learning-loop` | - | "Standard operating procedure for Protocol 128 Hardened Learning Loop (Scout -> Synthesize -> Audit -> Seal -> Persist)." | - |
| `/sanctuary-persist` | - | Protocol 128 Phase VI - Soul Persistence (Broadcast to Hugging Face) | - |
| `/sanctuary-protocol` | - | Manage Protocol Documents | - |
| `/sanctuary-retrospective` | 1 | Mandatory self-retrospective and continuous improvement check after completing any codify workflow. | /sanctuary-end |
| `/sanctuary-scout` | - | Protocol 128 Phase I - The Learning Scout (Debrief & Orientation) | - |
| `/sanctuary-seal` | - | Protocol 128 Phase V - The Technical Seal (Snapshot & Validation) | - |
| `/sanctuary-start` | 1 | Universal pre-flight and Spec initialization for all workflows. Determines work type and ensures Spec-Plan-Tasks exist. | - |
| `/tasks-manage` | - | Manage Maintenance Tasks (Kanban) | - |

## Quick Reference (All)

| Command | Track | Description |
| :--- | :--- | :--- |
| `/adr-manage` | Factory | Creates a new Architecture Decision Record (ADR) with proper numbering and template. |
| `/post-move-link-check` | Factory | Run link checker after moving or renaming files/folders |
| `/spec-kitty.analyze` | Discovery | Perform a non-destructive cross-artifact consistency and quality analysis across spec.md, plan.md, and tasks.md after task generation. |
| `/spec-kitty.checklist` | Discovery | Generate a custom checklist for the current feature based on user requirements. |
| `/spec-kitty.clarify` | Discovery | Identify underspecified areas in the current feature spec by asking up to 5 highly targeted clarification questions and encoding answers back into the spec. |
| `/spec-kitty.constitution` | Discovery | Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync. |
| `/spec-kitty.implement` | Discovery | Execute the implementation plan by processing and executing all tasks defined in tasks.md |
| `/spec-kitty.plan` | Discovery | Execute the implementation planning workflow using the plan template to generate design artifacts. |
| `/spec-kitty.specify` | Discovery | Create or update the feature specification from a natural language feature description. |
| `/spec-kitty.tasks` | Discovery | Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts. |
| `/spec-kitty.tasks-to-issues` | Discovery | Convert existing tasks into actionable, dependency-ordered GitHub issues for the feature based on available design artifacts. |
| `/adr-manage` | Factory | Manage Architecture Decision Records (ADR) |
| `/sanctuary-audit` | Factory | Protocol 128 Phase IV - Red Team Audit (Capture Snapshot) |
| `/bundle-manage` | Factory | Create a markdown bundle from a set of files using a manifest. |
| `/sanctuary-chronicle` | Factory | Manage Chronicle Entries (Journaling) |
| `/sanctuary-end` | Factory | Standard post-flight closure for all codify/investigate workflows. Handles human review, git commit, PR verification, and cleanup. |
| `/sanctuary-ingest` | Factory | Run RAG Ingestion (Protocol 128 Phase IX) |
| `/sanctuary-learning-loop` | Factory | "Standard operating procedure for Protocol 128 Hardened Learning Loop (Scout -> Synthesize -> Audit -> Seal -> Persist)." |
| `/sanctuary-persist` | Factory | Protocol 128 Phase VI - Soul Persistence (Broadcast to Hugging Face) |
| `/sanctuary-protocol` | Factory | Manage Protocol Documents |
| `/sanctuary-retrospective` | Factory | Mandatory self-retrospective and continuous improvement check after completing any codify workflow. |
| `/sanctuary-scout` | Factory | Protocol 128 Phase I - The Learning Scout (Debrief & Orientation) |
| `/sanctuary-seal` | Factory | Protocol 128 Phase V - The Technical Seal (Snapshot & Validation) |
| `/sanctuary-start` | Factory | Universal pre-flight and Spec initialization for all workflows. Determines work type and ensures Spec-Plan-Tasks exist. |
| `/tasks-manage` | Factory | Manage Maintenance Tasks (Kanban) |
| `/tool-inventory-manage` | Curate | Update tool inventories, RLM cache, and associated artifacts after creating or modifying tools. |
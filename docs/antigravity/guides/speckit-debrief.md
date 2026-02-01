# Spec-Kit Antigravity: Agent Debrief

## Overview
You have been upgraded with the **Spec-Kit Antigravity** toolset. This module introduces a rigorous, spec-driven development workflow designed to co-exist with your current capabilities. 

All new Spec-Kit commands are prefixed with `speckit-` to avoid collisions with your existing workflows (e.g., `codify-*`, `investigate-*`).

## Core Philosophy
**"Think before you code."**
The workflow enforces a linear progression:  
`Principles` → `Specification` → `Clarification` → `Architecture Plan` → `Task Breakdown` → `Implementation`

## Toolset & Commands

| Phase | Command | Purpose | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **0. Constitution** | `/speckit-constitution` | Establish project-wide governing principles. **Run this first.** | User Prompt | `.agent/rules/speckit-constitution.md` |
| **1. Specify** | `/speckit-specify` | Define *what* to build (functional requirements). | User Prose | `specs/XXX/spec.md` |
| **2. Clarify** | `/speckit-clarify` | Identify ambiguity and de-risk the spec before planning. | `spec.md` | Updates `spec.md` (Clarifications) |
| **3. Plan** | `/speckit-plan` | Define *how* to build it (architecture, stack, data models). | `spec.md`, Constitution | `specs/XXX/plan.md` |
| **4. Checklist** | `/speckit-checklist` | (Optional) Generate QA/Rules checklist for the feature. | `spec.md`, `plan.md` | `specs/XXX/checklist.md` |
| **5. Tasks** | `/speckit-tasks` | Break plan into atomic, actionable steps. | `plan.md` | `specs/XXX/tasks.md` |
| **6. Analyze** | `/speckit-analyze` | (Optional) Cross-artifact consistency check. | `spec.md`, `plan.md`, `tasks.md` | Analysis Report |
| **7. Implement** | `/speckit-implement` | Execute the tasks. | `tasks.md` | Code Changes |

## Directory Structure Changes

Files have been installed into `.agent/`:

```text
.agent/
├── rules/
│   └── speckit-constitution.md     <-- Project principles
├── templates/                      <-- Templates for artifacts
│   ├── plan-template.md
│   ├── spec-template.md
│   └── tasks-template.md
└── workflows/                      <-- Executable agent instructions
    ├── speckit-specify.md
    ├── speckit-plan.md
    └── ...
```

## Protocol for the Agent

1.  **Prefix Usage**: Always use the `speckit-` prefix for these core SDLC steps. Legacy commands like `/plan` should be considered deprecated or belonging to a different context.
2.  **Context Loading**: When running `/speckit-plan`, you **MUST** read `.agent/rules/speckit-constitution.md` to ensure architectural decisions align with project principles.
3.  **Artifact Location**: All generated artifacts (`spec.md`, `plan.md`, `tasks.md`) are stored in feature-specific directories under `specs/` (e.g., `specs/001-feature-name/`). Do NOT clutter the root directory.
4.  **Template Usage**: Use the templates in `.agent/templates/` as the source of truth for file structures.

## Immediate Next Steps for Agent
1.  Run `/speckit-constitution` (if not already done) to verify/create the project's ruling document.
2.  Suggest `/speckit-specify` to the user when they want to start a new feature or analysis module.

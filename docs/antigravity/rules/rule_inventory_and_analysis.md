# Rule Inventory & Analysis

**Objective**: Audit existing .agent/rules/, identify duplicates/obsoletes, and rightsize the constitution.

## Active Rules (Project Sanctuary)

| File Name (Linked) | Summary of Intent | Constitution Mapping | Notes |
| :--- | :--- | :--- | :--- |
| [[adr_creation_policy|adr_creation_policy.md]] | Defines when and how to create Architecture Decision Records (ADRs). | Code & Architecture | Critical for architectural decisions. |
| [[coding_conventions_policy|coding_conventions_policy.md]] | Coding standards for Python, JS, C#. | Code & Architecture | Standards for code quality. |
| [[dependency_management_policy|dependency_management_policy.md]] | Defines how to manage dependencies (pip, npm, NuGet). | Code & Architecture | Critical for supply chain security. |
| [[git_workflow_policy|git_workflow_policy.md]] | Defines branching strategy and commit standards. | Security Protocol (Iron Root) | Enforces "No Direct Commits to Main". |
| [[human_gate_policy|human_gate_policy.md]] | **The Supreme Law** (Zero Trust, Approval Gate, Emergency Stop). | Core Principles (I. Human Gate) | This IS the core of the Constitution. |
| [[spec_driven_development_policy|spec_driven_development_policy.md]] | Dual-Track Management (Spec-Driven Features + Kanban Maintenance). | Task Management | Defines Track A/B workflow. |
| [[tool_discovery_and_retrieval_policy|tool_discovery_and_retrieval_policy.md]] | "Late-Binding" protocol for tool discovery via RLM Cache. | Operations & Capabilities | Defines Agent Capabilities. |
| [[workflow_standardization_policy|workflow_standardization_policy.md]] | "Command-Driven Improvement" and Slash Command usage. | Operations & Capabilities | Defines "How we work". |
| [[progressive_elaboration_policy|progressive_elaboration_policy.md]] | "Living Documents" philosophy for iterative refinement. | Operations & Capabilities | Enables documentation evolution. |

## Archived/Merged Rules

| Original File | Disposition | Merged Into |
| :--- | :--- | :--- |
| `antigravity_command_policy.md` | **MERGED** | `workflow_standardization_policy.md` |
| `cli_tool_usage_policy.md` | **MERGED** | `tool_discovery_and_retrieval_policy.md` |
| `tool_inventory_policy.md` | **MERGED** | `tool_discovery_and_retrieval_policy.md` |
| `task_creation_policy.md` | **MERGED** | `spec_driven_development_policy.md` |

## Reference Documents

| Document | Purpose |
| :--- | :--- |
| [[constitution|Constitution]] | The supreme governing document for Project Sanctuary. |
| [[sanctuary-guardian-prompt|Sanctuary Guardian Prompt]] | 9-Phase Learning Loop and operational protocols. |
| [[cognitive_continuity_policy|Cognitive Continuity Policy]] | Memory rules loaded into LLM context. |

**Version**: 2.0 | **Updated**: 2026-01-31

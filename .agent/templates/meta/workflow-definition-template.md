# WF-NNNN: [Workflow Title]

## Metadata
*   **ID:** WF-NNNN
*   **Status:** [Active | Deprecated | Draft]
*   **Tier:** [1-Atomic | 2-Composite | 3-Orchestrated]
*   **Related ADRs:** [ADR-XXX, ADR-YYY]

## Description
[High-level description of the workflow. What is the goal? e.g. "To automate the learning loop seal and persistence process."]

## Visual Flow (Mermaid)
[View Workflow Diagram (MMD)](../diagrams/templates/workflow-template-example.mmd)

![Workflow Diagram](../diagrams/templates/workflow-template-example.png)

## Step-by-Step Workflow

### 1. Pre-Flight
*   **Trigger:** [What starts this? e.g. `/sanctuary-start`]
*   **Constitutional Check:** Verify against `.agent/rules/constitution.md`
*   **Action:** [Description]

### 2. Execution
*   **Tools:** [CLI commands or MCP tools used]
*   **Logic:** [Key logic or validations applied]
*   **Artifacts:** [Files created or modified]

### 3. Post-Flight
*   **Outcome:** [End state]
*   **Closure:** Run `/sanctuary-end` and `/sanctuary-retrospective`

## Key Rules & Validations
*   Human Gate: Requires explicit approval before state-changing actions.
*   Protocol 128: Must seal and persist learning before session end.

## Integration Points
*   **CLI:** `python tools/cli.py workflow start --name [name] --target [target]`
*   **Workflows:** References `/spec-kitty.*` or `/codify-*` as needed

# Command.json Schema for the Commandable Council

This document defines the JSON schema for `command.json`, the command interface used by the Guardian (Meta-Orchestrator) to issue tasks to the Autonomous Triad Council.

## Overview

The `command.json` file is the primary mechanism for commanding the persistent Council. It is placed in the `council_orchestrator/` directory and is automatically detected and consumed by the running orchestrator. Each command triggers a multi-round deliberation among the Coordinator, Strategist, and Auditor agents.

## JSON Structure

```json
{
  "task_description": "string",
  "input_artifacts": ["string"],
  "output_artifact_path": "string",
  "config": {
    "max_rounds": number
  }
}
```

## Field Definitions

### `task_description` (required, string)
- **Description**: A clear, high-level description of the strategic task to be performed by the Triad.
- **Purpose**: Defines the objective that the agents will deliberate on. Should be actionable and focused on analysis, synthesis, or strategic planning.
- **Examples**:
  - `"Review the previously generated 'triad_symposium_log.md' to assess the quality of our first autonomous conversation and propose one refinement to the orchestration logic."`
  - `"Perform a critical self-review of our own architecture. The goal is to identify the single greatest weakness in the current Orchestrator design (Protocols 94 & 95) and propose a concrete, actionable hardening to improve our resilience, efficiency, or autonomy."`

### `input_artifacts` (optional, array of strings)
- **Description**: An array of file paths (relative to the project root) that contain context or data the Triad should analyze.
- **Purpose**: Provides the agents with necessary input materials. Files are automatically injected into the conversation as "Initial knowledge provided" context.
- **Examples**:
  - `["WORK_IN_PROGRESS/API_NATIVE_PROBE_01/triad_symposium_log.md"]`
  - `["council_orchestrator/orchestrator_architecture_package.md"]`
- **Notes**: If no input artifacts are needed, this can be an empty array `[]` or omitted entirely.

### `output_artifact_path` (required, string)
- **Description**: The file path (relative to the project root) where the Triad's final synthesis will be saved.
- **Purpose**: Specifies the location for the generated artifact. The directory will be created if it doesn't exist.
- **Examples**:
  - `"WORK_IN_PROGRESS/COUNCIL_DIRECTIVES/directive_001_orchestrator_refinement.md"`
  - `"WORK_IN_PROGRESS/COUNCIL_DIRECTIVES/directive_002_orchestrator_self_hardening.md"`

### `config` (optional, object)
- **Description**: Configuration options for the deliberation process.
- **Fields**:
  - `max_rounds` (optional, number): The maximum number of deliberation rounds. Each round consists of one response from each agent (Coordinator, Strategist, Auditor). Defaults to 3 if not specified.
- **Examples**:
  - `{"max_rounds": 2}`
  - `{"max_rounds": 3}`

## Usage Workflow

1. **Craft the Command**: Create or edit `council_orchestrator/command.json` with the desired task parameters.
2. **Automatic Detection**: The running orchestrator detects the file within seconds.
3. **Execution**: The Triad deliberates for the specified number of rounds, potentially requesting additional files via the orchestrator.
4. **Artifact Generation**: The final synthesis is saved to the specified output path.
5. **Cleanup**: The `command.json` file is automatically deleted, and the orchestrator returns to idle monitoring.

## Best Practices

- **Task Clarity**: Ensure `task_description` is specific enough to guide the agents but broad enough for creative deliberation.
- **Artifact Management**: Use descriptive paths in `WORK_IN_PROGRESS/COUNCIL_DIRECTIVES/` for consistency.
- **Round Configuration**: Start with 2-3 rounds; increase only if the task requires deeper analysis.
- **File Paths**: Always use relative paths from the project root. The orchestrator validates file existence for input artifacts.

## Example Commands

### Example 1: Architecture Review
```json
{
  "task_description": "Review the previously generated 'triad_symposium_log.md' to assess the quality of our first autonomous conversation and propose one refinement to the orchestration logic.",
  "input_artifacts": [
    "WORK_IN_PROGRESS/API_NATIVE_PROBE_01/triad_symposium_log.md"
  ],
  "output_artifact_path": "WORK_IN_PROGRESS/COUNCIL_DIRECTIVES/directive_001_orchestrator_refinement.md",
  "config": {
    "max_rounds": 2
  }
}
```

### Example 2: Self-Improvement Task
```json
{
  "task_description": "Perform a critical self-review of our own architecture. The goal is to identify the single greatest weakness in the current Orchestrator design (Protocols 94 & 95) and propose a concrete, actionable hardening to improve our resilience, efficiency, or autonomy.",
  "input_artifacts": [
    "council_orchestrator/orchestrator_architecture_package.md"
  ],
  "output_artifact_path": "WORK_IN_PROGRESS/COUNCIL_DIRECTIVES/directive_002_orchestrator_self_hardening.md",
  "config": {
    "max_rounds": 3
  }
}
```

This schema ensures consistent, reliable communication between the Guardian and the Autonomous Council.
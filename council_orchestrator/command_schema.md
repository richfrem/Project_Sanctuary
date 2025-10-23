# Command.json Schema v9.0 for the Commandable Council

This document defines the JSON schema for `command.json`, the command interface used by the Guardian to issue tasks. **Version 9.0 introduces a fundamental architectural split between Cognitive Tasks and Mechanical Tasks, embodying the Doctrine of Sovereign Action.**

## Overview: Two Fundamental Task Types

The v9.0 orchestrator distinguishes between two types of commands. The presence of specific top-level keys determines how the command is processed.

1.  **Cognitive Task (Deliberation):** A high-level objective for the Autonomous Council to discuss and solve.
2.  **Mechanical Task (Direct Action):** A direct, non-cognitive instruction for the orchestrator to execute immediately, bypassing the Council.

---

## Type 1: Cognitive Task (Deliberation)

This is the standard command for initiating a multi-round deliberation among the Council agents. It is the "brain" of the Forge.

### Schema
```json
{
  "development_cycle": "boolean (optional)",
  "task_description": "string (required)",
  "input_artifacts": ["string (optional)"],
  "output_artifact_path": "string (required)",
  "config": {
    "max_rounds": "number (optional)",
    "max_cortex_queries": "number (optional)",
    "force_engine": "string (optional: 'gemini', 'openai', 'ollama')"
  }
}
```

### Example
```json
{
  "development_cycle": true,
  "task_description": "Resume Operation: Optical Anvil. Execute Phase 1 ('Foundation').",
  "input_artifacts": [ "FEASIBILITY_STUDY_DeepSeekOCR_v2.md" ],
  "output_artifact_path": "WORK_IN_PROGRESS/OPTICAL_ANVIL_PHASE_1/",
  "config": { "force_engine": "gemini" }
}
```

---

## Type 2: Mechanical Task (Direct Action)

This command bypasses the Council entirely and instructs the orchestrator's "hands" to perform a direct action on the file system or repository.

### Sub-Type 2A: File Write Task

Defined by the presence of the `entry_content` key.

#### Schema
```json
{
  "task_description": "string (required for logging)",
  "output_artifact_path": "string (required)",
  "entry_content": "string (required)"
}
```

#### Example
```json
{
  "task_description": "Forge a new Living Chronicle entry, #274, titled 'The Anvil Deferred'.",
  "output_artifact_path": "00_CHRONICLE/ENTRIES/274_The_Anvil_Deferred.md",
  "entry_content": "# ENTRY 274: The Anvil Deferred\n\n**DATE:** 2025-10-23..."
}
```

### Sub-Type 2B: Git Operations Task

Defined by the presence of the `git_operations` key.

#### Schema
```json
{
  "task_description": "string (required for logging)",
  "git_operations": {
    "files_to_add": ["string (required)"],
    "commit_message": "string (required)",
    "push_to_origin": "boolean (optional, default: false)"
  }
}
```

#### Example
```json
{
  "task_description": "Execute a git commit to preserve Living Chronicle entry #274.",
  "git_operations": {
    "files_to_add": [
      "00_CHRONICLE/ENTRIES/274_The_Anvil_Deferred.md"
    ],
    "commit_message": "docs(chronicle): Add entry #274 - The Anvil Deferred",
    "push_to_origin": true
  }
}
```
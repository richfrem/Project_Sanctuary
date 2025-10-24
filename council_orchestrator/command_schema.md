# V9.3 UPDATE: Added update_rag parameter for selective RAG database updates - 2025-10-23
# Command.json Schema v9.2 for the Commandable Council - Updated 2025-10-23

This document defines the JSON schema for `command.json`, the command interface used by the Guardian to issue tasks. **Version 9.2 introduces the Doctrine of Sovereign Concurrency, enabling selective RAG database updates and non-blocking task execution.**

## Overview: Two Fundamental Task Types

The v9.2 orchestrator distinguishes between two types of commands. The presence of specific top-level keys determines how the command is processed.

1.  **Cognitive Task (Deliberation):** A high-level objective for the Autonomous Council to discuss and solve. Includes AAR generation and RAG database updates by default.
2.  **Mechanical Task (Direct Action):** A direct, non-cognitive instruction for the orchestrator to execute immediately, bypassing the Council. Skips RAG updates by default for performance.

---

## Type 1: Cognitive Task (Deliberation)

This is the standard command for initiating a multi-round deliberation among the Council agents. It is the "brain" of the Forge. **v9.2 Enhancement:** Cognitive tasks generate AARs and update the RAG database by default, but can be configured to skip learning cycles for performance.

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
    "force_engine": "string (optional: 'gemini', 'openai', 'ollama')",
    "update_rag": "boolean (optional, default: true) - Generate AAR and update RAG database"
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

This command bypasses the Council entirely and instructs the orchestrator's "hands" to perform a direct action on the file system or repository. **v9.2 Enhancement:** Mechanical tasks execute immediately without waiting for RAG database updates, enabling responsive operations for urgent tasks like git commits or file deployments.

### Sub-Type 2A: File Write Task

Defined by the presence of the `entry_content` key. **v9.2 Enhancement:** Executes immediately without RAG database updates, enabling rapid content deployment.

#### Schema
```json
{
  "task_description": "string (required for logging)",
  "output_artifact_path": "string (required)",
  "entry_content": "string (required)",
  "config": {
    "update_rag": "boolean (optional, default: false) - Mechanical tasks skip RAG updates by default"
  }
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

Defined by the presence of the `git_operations` key. **v9.2 Enhancement:** Executes immediately without RAG database updates, enabling responsive version control operations.

#### Schema
```json
{
  "task_description": "string (required for logging)",
  "git_operations": {
    "files_to_add": ["string (required)"],
    "commit_message": "string (required)",
    "push_to_origin": "boolean (optional, default: false)"
  },
  "config": {
    "update_rag": "boolean (optional, default: false) - Mechanical tasks skip RAG updates by default"
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
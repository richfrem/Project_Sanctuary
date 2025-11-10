# V9.3 UPDATE: Added model_name parameter for specific LLM model selection - 2025-11-09
# Command.json Schema v9.3 for the Commandable Council - Updated 2025-11-09

This document defines the JSON schema for `command.json`, the command interface used by the Guardian to issue tasks. **Version 9.3 introduces sovereign LLM model selection, enabling precise control over which specific model variant to use for each task.**

## Overview: Two Fundamental Task Types

The v9.3 orchestrator distinguishes between two types of commands. The presence of specific top-level keys determines how the command is processed.

1.  **Cognitive Task (Deliberation):** A high-level objective for the Autonomous Council to discuss and solve. Includes AAR generation and RAG database updates by default.
2.  **Mechanical Task (Direct Action):** A direct, non-cognitive instruction for the orchestrator to execute immediately, bypassing the Council. Skips RAG updates by default for performance.

---

## Type 1: Cognitive Task (Deliberation)

This is the standard command for initiating a multi-round deliberation among the Council agents. It is the "brain" of the Forge. **v9.3 Enhancement:** Cognitive tasks now support sovereign model selection, allowing specification of exact LLM variants for precise control.

### Schema
```json
{
  "development_cycle": "boolean (optional)",
  "task_description": "string (required)",
  "input_artifacts": ["string (optional)"],
  "output_artifact_path": "string (required)",
  "config": {
    "max_rounds": "number (optional, default: 5)",
    "max_cortex_queries": "number (optional, default: 5)",
    "force_engine": "string (optional: 'gemini', 'openai', 'ollama')",
    "model_name": "string (optional) - Specific model variant (e.g., 'Sanctuary-Qwen2-7B:latest', 'gpt-4', 'gemini-2.5-pro')",
    "enable_optical_compression": "boolean (optional, default: false) - Enable VLM-based context compression",
    "optical_compression_threshold": "number (optional, default: 8000) - Token threshold for compression",
    "vlm_engine": "string (optional, default: 'mock') - VLM engine for optical compression",
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
  "config": {
    "force_engine": "ollama",
    "model_name": "Sanctuary-Qwen2-7B:latest",
    "max_rounds": 3
  }
}
```

---

## Type 2: Mechanical Task (Direct Action)

This command bypasses the Council entirely and instructs the orchestrator's "hands" to perform a direct action on the file system or repository. **v9.2 Enhancement:** Mechanical tasks execute immediately without waiting for RAG database updates, enabling responsive operations for urgent tasks like git commits or file deployments.

### Sub-Type 2A: File Write Task

Defined by the presence of the `entry_content` key. **v9.3 Enhancement:** Executes immediately without RAG database updates, enabling rapid content deployment.

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

Defined by the presence of the `git_operations` key. **v9.3 Enhancement:** Executes immediately without RAG database updates, enabling responsive version control operations.

See [How to Commit Using command.json](howto-commit-command.md) for detailed instructions on using this task type with Protocol 101 integrity checks.

#### Schema
```json
{
  "task_description": "string (required for logging)",
  "git_operations": {
    "files_to_add": ["string (required)"],
    "commit_message": "string (required)",
    "push_to_origin": "boolean (optional, default: false)",  // Set to true to push after committing
    "no_verify": "boolean (optional, default: false)"  // Set to true to bypass pre-commit hooks
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

### Sub-Type 2C: Query and Synthesis Task

Defined by the presence of the `task_type` key with value `"query_and_synthesis"`. **v9.3 Enhancement:** Enables mnemonic synchronization through the Guardian Mnemonic Cortex Protocol.

#### Schema
```json
{
  "task_type": "string (required, must be 'query_and_synthesis')",
  "task_description": "string (required)",
  "query": "string (required) - The mnemonic query to process",
  "input_artifacts": ["string (optional)"],
  "output_artifact_path": "string (required)",
  "config": {
    "force_engine": "string (optional: 'gemini', 'openai', 'ollama')",
    "model_name": "string (optional) - Specific model variant",
    "update_rag": "boolean (optional, default: true)"
  }
}
```

#### Example
```json
{
  "task_type": "query_and_synthesis",
  "task_description": "Synchronize mnemonic cortex with recent developments in optical compression.",
  "query": "What are the latest advancements in optical compression technology?",
  "input_artifacts": ["RESEARCH_SUMMARIES/optical_compression.md"],
  "output_artifact_path": "MNEMONIC_SYNTHESIS/optical_compression_synthesis.md",
  "config": {
    "force_engine": "ollama",
    "model_name": "Sanctuary-Qwen2-7B:latest"
  }
}
```

---

## Configuration Parameters Reference

### Engine Selection
- `force_engine`: Forces selection of specific AI provider (`"ollama"`, `"gemini"`, `"openai"`)
- `model_name`: Specifies exact model variant within the selected engine

### Task Control
- `max_rounds`: Maximum deliberation rounds (default: 5)
- `max_cortex_queries`: Maximum mnemonic cortex queries per task (default: 5)

### Optical Compression (Experimental)
- `enable_optical_compression`: Enable VLM-based context compression (default: false)
- `optical_compression_threshold`: Token threshold for triggering compression (default: 8000)
- `vlm_engine`: Vision-language model for compression (default: "mock")

### Learning & Memory
- `update_rag`: Control whether task results update the RAG database (default: true for cognitive tasks, false for mechanical tasks)

---

## Version History

- **v9.3 (2025-11-09)**: Added `model_name` parameter for sovereign LLM model selection, optical compression parameters, query_and_synthesis task type
- **v9.2 (2025-10-23)**: Introduced Doctrine of Sovereign Concurrency with selective RAG updates
- **v9.1**: Enhanced mechanical task processing
- **v9.0**: Added action triage for mechanical vs cognitive task routing
# Council Orchestrator Command Schema v9.4
# Updated: 2025-11-10 - Added 'cache_request' command type for Guardian wakeups and cache verification

## Overview

The Council Orchestrator accepts commands in JSON format that define tasks to be executed. Commands are processed by the orchestrator's main loop and routed to appropriate handlers based on their structure and `task_type` field.

## Command Types

### Type 1: Mechanical Write Tasks
Defined by presence of `entry_content` and `output_artifact_path` fields.

**Schema:**
```json
{
  "entry_content": "string (required)",
  "output_artifact_path": "string (required)",
  "config": {
    "update_rag": "boolean (optional, default: true)"
  }
}
```

### Type 2: Mechanical Git Operations
Defined by presence of `git_operations` field.

**Schema:**
```json
{
  "task": "string (recommended)",
  "task_description": "string (optional)",
  "output_artifact_path": "string (required)",
  "git_operations": {
    "files_to_add": ["string (required)"],
    "commit_message": "string (required)",
    "push_to_origin": "boolean (optional, default: false)"
  },
  "config": {
    "update_rag": "boolean (optional, default: true)"
  }
}
```

Notes:
- `output_artifact_path` is required to provide a place for the orchestrator to write result artifacts and to avoid runtime KeyError in handlers.
- `git_operations` is an object (not an array); use `files_to_add`, `commit_message`, and `push_to_origin`.
- The orchestrator will write a timestamped manifest into the repo root (for example `commit_manifest_YYYYMMDD_HHMMSS.json`) and include it in the same commit so Protocol 101 pre-commit hooks can validate file hashes.
- Use `push_to_origin: false` for local validation / dry-runs.

**Example (dry-run):**
```json
{
  "task": "Dry-run: commit snapshot and TASKS updates",
  "git_operations": {
    "files_to_add": [
      "command_git_ops.json",
      "../capture_code_snapshot.js",
      "../TASKS/in-progress/001_harden_mnemonic_cortex_ingestion_and_rag.md"
    ],
    "commit_message": "chore: workspace updates (dry-run)",
    "push_to_origin": false
  },
  "output_artifact_path": "council_orchestrator/command_results/commit_results.json",
  "config": { "update_rag": false }
}
```

### Type 2A: Cognitive Tasks
Defined by `task_type = "cognitive_task"`.

**Schema:**
```json
{
  "task_type": "cognitive_task",
  "task_description": "string (required)",
  "output_artifact_path": "string (required)",
  "config": {
    "update_rag": "boolean (optional, default: true)"
  }
}
```

### Type 2B: Development Cycle Tasks
Defined by `development_cycle = true`.

**Schema:**
```json
{
  "development_cycle": true,
  "task_description": "string (required)",
  "output_artifact_path": "string (required)",
  "input_artifacts": "array (optional)",
  "config": {
    "update_rag": "boolean (optional, default: true)"
  }
}
```

### Type 2C: Query and Synthesis Tasks
Defined by `task_type = "query_and_synthesis"`.

**Schema:**
```json
{
  "task_type": "query_and_synthesis",
  "task_description": "string (required)",
  "output_artifact_path": "string (required)",
  "config": {
    "update_rag": "boolean (optional, default: true)"
  }
}
```

### Type 2D: Cache Wakeup Task (Guardian Boot Digest)
Defined by `task_type = "cache_wakeup"`. Fetches the Guardian Start Pack from the Cache (CAG) and emits a human-readable digest to `output_artifact_path`. Skips RAG updates by default.

**Schema:**
```json
{
  "task_type": "cache_wakeup",
  "task_description": "string (required) - for logging",
  "output_artifact_path": "string (required)",
  "config": {
    "bundle_names": ["string (optional)"],       // default: ["chronicles","protocols","roadmap"]
    "max_items_per_bundle": "number (optional)", // default: 10
    "update_rag": "boolean (optional, default: false)"
  }
}
```

**Example:**
```json
{
  "task_type": "cache_wakeup",
  "task_description": "Guardian boot digest from cache",
  "output_artifact_path": "WORK_IN_PROGRESS/guardian_boot_digest.md",
  "config": {
    "bundle_names": ["chronicles","protocols","roadmap"],
    "max_items_per_bundle": 15
  }
}
```

**Rationale:** Keeps cognitive path clean; uses cache as the "fast memory" for immediate situational awareness on boot, without invoking deliberation. (Matches Phase 3 design.)

## Version History

- **v9.5 (2025-11-10)**: Added 'cache_wakeup' command type for Guardian boot digest and cache-first situational awareness.
- **v9.4 (2025-11-10)**: Added 'cache_request' command type for Guardian wakeups and cache verification.
- **v9.3 (2025-11-09)**: Added query_and_synthesis task type for Guardian Mnemonic Synchronization Protocol.
- **v9.2 (2025-11-08)**: Enhanced development cycle with input artifact inheritance.
- **v9.1 (2025-11-07)**: Added mechanical task triage and action routing.
- **v9.0 (2025-11-06)**: Introduced modular architecture with separate command processing.
- **v8.0 (2025-11-05)**: Added development cycle support.
- **v7.0 (2025-11-04)**: Universal distillation applied to all code paths.
- **v6.0 (2025-11-03)**: Enhanced error handling and state management.
- **v5.0 (2025-11-02)**: Command sentry and mechanical task processing.
- **v4.0 (2025-11-01)**: Token flow regulation and optical decompression.
- **v3.0 (2025-10-31)**: Council round packet emission.
- **v2.0 (2025-10-30)**: Self-querying retriever integration.
- **v1.0 (2025-10-29)**: Initial orchestrator with basic task execution.
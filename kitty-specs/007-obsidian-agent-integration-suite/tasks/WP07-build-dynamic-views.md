---
work_package_id: "WP07"
title: "Build Obsidian Dynamic Views Skills (Bases & Canvas)"
lane: "planned"
dependencies: ["WP01", "WP06"]
subtasks: ["T030", "T031", "T032", "T033", "T034"]
---

# Work Package Prompt: WP07 – Build Obsidian Dynamic Views Skills (Bases & Canvas)

## Objectives & Success Criteria
- Programmatic read/write access to DB-like `.base` views and `.canvas` flowcharts.
- Robust schema validation ensuring the agent doesn't corrupt dynamic files.

## Context & Constraints
- These structures are highly proprietary. Degrade gracefully if schemas evolve unexpectedly.

## Subtasks & Detailed Guidance

### Subtask T030 – Scaffold Plugin Frameworks
- **Purpose**: Prepare architectures.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-bases-manager/` and `plugins/obsidian-integration/skills/obsidian-canvas-architect/` alongside their corresponding `SKILL.md` documents.

### Subtask T031 – `.base` Table Manipulation
- **Purpose**: Interact with dashboard state.
- **Steps**: Write logic decoding the YAML structure native to `Obsidian Bases`. Build functions allowing an agent to confidently append or update row data while leaving view configurations untouched.

### Subtask T032 – JSON Canvas Specifications
- **Purpose**: Visual flowchart mapping.
- **Steps**: Build a client utilizing `JSON Canvas Spec 1.0`. Structure logic to programmatically place semantic `nodes` and connect them with directional `edges`.

### Subtask T033 – Graceful Error Degradation
- **Purpose**: System stability.
- **Steps**: Ensure any `KeyError` or schema mismatch caught when loading these files issues a clean API warning back to the agent rather than crashing the loop or wiping the file.

### Subtask T034 – Schema Verifications
- **Purpose**: Safety.
- **Steps**: Write verification unit tests mapping against synthetic `.base` files and mock `.canvas` boards.

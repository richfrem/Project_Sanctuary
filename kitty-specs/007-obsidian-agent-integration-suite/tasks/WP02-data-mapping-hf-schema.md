---
work_package_id: "WP02"
subtasks:
  - "T005"
  - "T006"
  - "T007"
  - "T008"
  - "T009"
title: "Research Data Mapping to HF Schema"
phase: "Phase 0 - Research"
lane: "planned"  
assignee: ""      
agent: ""         
shell_pid: ""     
review_status: "" 
reviewed_by: ""   
dependencies: []
history:
  - timestamp: "2026-02-26T16:15:00Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
---

# Work Package Prompt: WP02 – Research Data Mapping to HF Schema

## ⚠️ IMPORTANT: Review Feedback Status

**Read this first if you are implementing this task!**
- **Has review feedback?**: Check the `review_status` field above. If it says `has_feedback`, scroll to the **Review Feedback** section immediately (right below this notice).

---

## Objectives & Success Criteria
- Define mapping rules avoiding strict directory syncing in favor of `source_path` attributes.
- Author an ADR formalizing mapping structures according to `HF_JSONL` specifications (ADR 081).
- Ignore attachments cleanly.

## Subtasks & Detailed Guidance

### Subtask T005 – Analyze HF soul_traces.jsonl schema
- **Purpose**: Baseline mapping schema against existing HF targets.
- **Steps**: Retrieve HF JSONL mapping schemas defined in ADR 081. Note JSON fields like `id`, `sha256`, `timestamp`, `content`.

### Subtask T006 – Define mapping rules for nested folders
- **Purpose**: How directory nesting applies to JSON payload string fields.
- **Steps**: Establish explicit transformation instructions to flatten filesystem hierarchy properties directly into the `source_path` metadata column within the target JSONL file.

### Subtask T007 – Formalize the image/binary ignore rule
- **Purpose**: Keep the dataset pure and textual.
- **Steps**: Solidify the requirement that all binaries and images within Obsidian are filtered from JSON.

### Subtask T008 – Draft ADR
- **Purpose**: Create physical documentation of data mapping decisions.
- **Steps**: Distill T005-T007 into an official Architectural Decision Record.

### Subtask T009 – Obtain human steward approval
- **Purpose**: Final sign-off.
- **Steps**: Submit the ADR for review and block on formal clearance.

## Activity Log

- 2026-02-26T16:15:00Z – system – lane=planned – Prompt created

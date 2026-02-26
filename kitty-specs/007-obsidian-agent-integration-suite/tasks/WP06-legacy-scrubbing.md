---
work_package_id: "WP06"
subtasks:
  - "T025"
  - "T026"
  - "T027"
  - "T028"
title: "Legacy Scrubbing & Automated Link Refactoring"
phase: "Phase 2 - Cleanup"
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

# Work Package Prompt: WP06 – Legacy Scrubbing & Automated Link Refactoring

## ⚠️ IMPORTANT: Review Feedback Status

**Read this first if you are implementing this task!**
- **Has review feedback?**: Check the `review_status` field above. If it says `has_feedback`, scroll to the **Review Feedback** section immediately (right below this notice).

---

## Objectives & Success Criteria
- Establish graph cohesiveness by mass-migrating relative reference strings into valid format wikilinks.
- Hard delete all legacy references pointing to "MCP" design semantics.
- Return zero failures upon grep checks on core repositories.

## Context & Constraints
- Only mutate `01_PROTOCOLS/` and `02_LEARNING/` for the link migrations to avoid breaking upstream pipeline files elsewhere.
- Remove outdated language from foundational system prompts (`.agent/`, `plugins/guardian-onboarding/`).

## Subtasks & Detailed Guidance

### Subtask T025 – Develop relative to wikilink converter script
- **Purpose**: Utility logic.
- **Steps**: Write a Python parser utilizing regex to identify `[Title](../path/file.md)` patterns inside markdown context and emit `[[file]]` replacements. Ensure it strips `.md` correctly.

### Subtask T026 – Apply refactoring
- **Purpose**: Graph realization.
- **Steps**: Execute parser across all nested files uniquely belonging to `.agent/learning/` (e.g., `01_PROTOCOLS` and `02_LEARNING` directories in the repository).

### Subtask T027 – Scrub MCP references
- **Purpose**: Enforce agent skill terminology.
- **Steps**: Traverse `sanctuary-guardian-prompt.md` and related `.agent/` and `plugins/guardian-onboarding/` files to systematically delete the term "MCP" or replace it dynamically with "Agent Skills" or "Plugins".

### Subtask T028 – Verify clean results
- **Purpose**: Hard validation gating.
- **Steps**: Build regression tests asserting `grep -i "mcp "` returns a non-zero exit code (i.e. zero lines matched) in the protected directories.

## Activity Log

- 2026-02-26T16:15:00Z – system – lane=planned – Prompt created

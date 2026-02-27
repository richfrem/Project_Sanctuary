---
work_package_id: WP04
title: Legacy Scrubbing & Automated Link Refactoring
lane: "doing"
dependencies: []  # Independent execution is intentional
base_branch: main
base_commit: 235fa6ef83bdab384cf33db2f99c0d31586df231
created_at: '2026-02-27T21:52:47.194250+00:00'
subtasks: [T016, T017, T018, T019]
shell_pid: "62896"
---

# Work Package Prompt: WP04 – Legacy Scrubbing & Automated Link Refactoring

## Objectives & Success Criteria
- Ensure zero Legacy internal relative links (`[Link](../../like-this.md)`) remain in the learning databanks.
- Purge legacy term "MCP architecture" acting as noise in context prompts.

## Context & Constraints
- Link transformation must happen early to create a clean testing graph for WP08.
- Must execute safely via a Dry-Run first.

## Subtasks & Detailed Guidance

### Subtask T016 – Develop refactoring script
- **Purpose**: Safe and verifiable substitution.
- **Steps**: Write a Python script targeting the regex pattern of markdown relative links. It MUST implement a dry-run flag (`--dry-run`) printing proposed changes without executing them. Exclude text inside triple-backtick code fences.

### Subtask T017 – Apply refactoring
- **Purpose**: Actual execution.
- **Steps**: Run the script against the `01_PROTOCOLS/` and `02_LEARNING/` directories. Commit the changes cleanly.

### Subtask T018 – Scrub "MCP" references
- **Purpose**: Remove confusing vocabulary.
- **Steps**: Hunt down "MCP" and "Model Context Protocol" occurrences in `sanctuary-guardian-prompt.md` and related context boot files. Replace them with "Agent Plugin Integration".

### Subtask T019 – Verify clean grep
- **Purpose**: Prove completion.
- **Steps**: Execute exhaustive greps proving neither standard relative markdown links nor "MCP" references remain in target paths.

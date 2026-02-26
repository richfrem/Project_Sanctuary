---
work_package_id: WP03
title: Build Obsidian CRUD Skill
lane: planned
dependencies: []
subtasks:
- T010
- T011
- T012
- T013
phase: Phase 1 - Implementation
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-02-26T16:15:00Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt generated via /spec-kitty.tasks
---

# Work Package Prompt: WP03 – Build Obsidian CRUD Skill

## ⚠️ IMPORTANT: Review Feedback Status

**Read this first if you are implementing this task!**
- **Has review feedback?**: Check the `review_status` field above. If it says `has_feedback`, scroll to the **Review Feedback** section immediately (right below this notice).

---

## Objectives & Success Criteria
- Implement fundamental Obsidian vault read/write operations via an Agent Skill.
- Note creation, retrieval, and append updates must function autonomously without manual human interaction.
- Passes explicit local CRUD lifecycle unit tests.

## Context & Constraints
- Must align strictly with the implementation pattern decided upon in the WP01 ADR.

## Subtasks & Detailed Guidance

### Subtask T010 – Create obsidian-crud plugin directory
- **Purpose**: Establish project plugin environment.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-crud/` directory and bootstrap standard `SKILL.md` template structure adhering to sanctuary guidelines.

### Subtask T011 – Implement read mechanisms
- **Purpose**: Retrieve vault markdown files dynamically.
- **Steps**: Implement retrieval scripts conforming to the WP01 architecture choice. Support fetching exact strings, frontmatter fields, and folder discovery.

### Subtask T012 – Implement create/update mechanisms
- **Purpose**: Edit note material directly from Agent environment.
- **Steps**: Create write scripts capable of adding or appending text logic inside target vault markdown notes, managing frontmatter securely. Ensure idempotency where necessary.

### Subtask T013 – Write unit verification tests
- **Purpose**: Automated regression coverage.
- **Steps**: Create unit test files that mimic lifecycle states: init note -> read note -> modify note -> read updated note -> assert results -> teardown.

## Activity Log

- 2026-02-26T16:15:00Z – system – lane=planned – Prompt created

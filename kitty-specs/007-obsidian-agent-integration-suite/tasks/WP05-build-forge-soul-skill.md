---
work_package_id: WP05
title: Build 'Forge Soul' Semantic Exporter Skill
lane: planned
dependencies: []
subtasks:
- T019
- T020
- T021
- T022
- T023
- T024
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

# Work Package Prompt: WP05 – Build 'Forge Soul' Semantic Exporter Skill

## ⚠️ IMPORTANT: Review Feedback Status

**Read this first if you are implementing this task!**
- **Has review feedback?**: Check the `review_status` field above. If it says `has_feedback`, scroll to the **Review Feedback** section immediately (right below this notice).

---

## Objectives & Success Criteria
- Push verified Obsidian contextual data (Soul Traces) exclusively into Hugging Face `soul_traces.jsonl`.
- Must firmly deny state-change execution if uncommitted Git drift exists (Protocol 101 execution lock).
- Must drop all binary attributes and successfully compile JSON payloads according to WP02 structure rules.

## Context & Constraints
- Constitution rules mandate Zero Trust Git. The script must execute a `git status` parity lock before touching remote layers.
- See ADR 081 for JSONL expected schema.

## Subtasks & Detailed Guidance

### Subtask T019 – Create forge-soul plugin directory
- **Purpose**: Establish plugin boundary.
- **Steps**: Scaffold `plugins/obsidian-integration/skills/forge-soul/` with an instructive `SKILL.md`.

### Subtask T020 – Implement Git Pre-Flight Check (Protocol 101) logic
- **Purpose**: Zero Trust execution block.
- **Steps**: Inject pre-flight validation logic to evaluate the working directory using `git status`. If there are uncommitted changes, abort process automatically and flag an error.

### Subtask T021 – Implement logic to identify sealed notes
- **Purpose**: Only export approved memories.
- **Steps**: Traverse the local vault explicitly searching for files with contextual approval frontmatter blocks (e.g., `status: sealed`, `#ADR`). Skip all draft or WIP notes.

### Subtask T022 – Implement data transformation logic
- **Purpose**: Adhere to HF constraints.
- **Steps**: Compile parsed files into dictionaries matching `id`, `sha256`, `timestamp`, `content`. Drop ALL attachment code (e.g. `![[image.png]]`) cleanly from text blobs, and map directories via `source_path` as strictly established in WP02.

### Subtask T023 – Implement export/append
- **Purpose**: Final network sync.
- **Steps**: Take transformed dictionaries and write locally to `.jsonl` then leverage the huggingface_hub api to push an updated version of the dataset safely.

### Subtask T024 – Write tests validating schema and Git blocks
- **Purpose**: Prove the gate works.
- **Steps**: Create test mock directories. Verify JSON payload structures via unit testing. Create a test executing the script in an artificially "dirty" git directory and assert it throws the defined abort exception.

## Activity Log

- 2026-02-26T16:15:00Z – system – lane=planned – Prompt created

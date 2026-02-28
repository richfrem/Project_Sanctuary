---
work_package_id: WP03
title: Research Data Mapping to HF Schema
lane: "done"
dependencies: []
base_branch: main
base_commit: e0f38e5a4133dcaa7ff1ff0823b9feb6a49a4aca
created_at: '2026-02-27T21:42:14.047135+00:00'
subtasks: [T011, T012, T013, T014, T015]
shell_pid: "62896"
agent: "Antigravity"
reviewed_by: "richfrem"
review_status: "approved"
---

# Work Package Prompt: WP03 – Research Data Mapping to HF Schema

## Objectives & Success Criteria
- Resolve how highly-nested markdown vaults map to `soul_traces.jsonl`.
- Provide an approved ADR standardizing the data translation.

## Context & Constraints
- Must adhere strictly to the `HF_JSONL` schema detailed in ADR 081.

## Subtasks & Detailed Guidance

### Subtask T011 – Analyze HF `soul_traces.jsonl` schema
- **Purpose**: Standardize integration target.
- **Steps**: Pull ADR 081 into context. Note constraints around JSON fields such as `content`, `domain`, and `timestamp`.

### Subtask T012 – Define folder mapping
- **Purpose**: Translate hierarchy to tags.
- **Steps**: An Obsidian vault has infinite directory depths. Define exact mapping rules on how nested folders project down into the JSONL `source_path` array.

### Subtask T013 – Formalize Attachment Rules
- **Purpose**: Protect the semantic vector spaces from huge binaries.
- **Steps**: Create explicit code filtering rules dictating that images, `.pdf`, `.mp4`, etc., are strictly ignored by the exporter.

### Subtask T014 – Draft ADR
- **Purpose**: Document the data mapping definitions.
- **Steps**: Generate the new Architectural Decision Record spanning the rules established in T012 and T013.

### Subtask T015 – Obtain human steward approval
- **Purpose**: Gate progress.
- **Steps**: Request human review for the ADR.

## Activity Log

- 2026-02-27T21:42:14Z – Antigravity – shell_pid=62896 – lane=doing – Assigned agent via workflow command
- 2026-02-27T21:48:08Z – Antigravity – shell_pid=62896 – lane=for_review – Completed HF schema mapping and ADR 100
- 2026-02-27T21:49:01Z – Antigravity – shell_pid=62896 – lane=for_review – Completed HF schema mapping and ADR 100 with user clarification on uni-directional export
- 2026-02-28T00:10:49Z – Antigravity – shell_pid=62896 – lane=done – Review passed: verified git diff, code quality, and separation of concerns

---
work_package_id: WP10
title: Phase 1.5 Integration & Synthetic Edge-Case Testing
lane: "done"
dependencies: []
base_branch: main
base_commit: da97dddc8356edf3ef0db1456c04c9dbfdc61a28
created_at: '2026-02-27T23:18:47.371767+00:00'
subtasks: [T046, T047, T048, T049, T050]
shell_pid: "55083"
agent: "Antigravity"
reviewed_by: "richfrem"
review_status: "approved"
---

# Work Package Prompt: WP10 – Phase 1.5 Integration & Synthetic Edge-Case Testing

## Objectives & Success Criteria
- Provide end-to-end integration safety.
- Expose the assembled `obsidian-integration` plugin suite to 100+ highly varied margin-case notes to prove the parser and CRUD engines won't crash.

## Context & Constraints
- Due to the risk of silent corruption on a real vault, these integration tests MUST run against an isolated `Synthetic Vault` instantiated in `/tmp` natively through Python Pytest fixtures.

## Subtasks & Detailed Guidance

### Subtask T046 – Spin Up Synthetic Vault
- **Purpose**: A safe sandbox.
- **Steps**: Write a Pytest fixture that creates a temporary directory mirroring an Obsidian vault structure.

### Subtask T047 – Seed Edge Casing Notes
- **Purpose**: Stress test the parsing engines.
- **Steps**: Programmatically generate notes featuring:
   - Malformed YAML (unquoted colons).
   - Massive strings with thousands of sequential Wikilinks.
   - Deeply nested directories (`path/to/very/deep/folder/note.md`).
   - `.canvas` nodes connected to non-existent nodes.
   - Standard text mixed heavily with triple-backtick bash output containing `[[` strings to ensure they are excluded from linking.

### Subtask T048 – Concurrent I/O Simulation
- **Purpose**: Stress test the WP06 atomic `.agent-lock` strategy.
- **Steps**: Unleash 10 asynchronous threads attempting to simultaneously update, link, and traverse the synthetic vault notes using the `obsidian-vault-crud` and `obsidian-graph-traversal` tools. Add a background thread that simulates Obsidian desktop behavior (periodically reading and rewriting a note file mimicking auto-save) while agent threads attempt concurrent CRUD operations. Assert that mtime detection catches every simulated conflict and no silent overwrites occur.

### Subtask T049 – Dry Run Forge Soul Export
- **Purpose**: Complete the pipeline.
- **Steps**: Take the final synthetic state. Trigger the semantic export pipeline in "dry-run" mode (writing JSONL to disk rather than network) and validate against the schema.

### Subtask T050 – Analyze Code Coverage and Error Flags
- **Purpose**: Confirm safety.
- **Steps**: Collect coverage metrics. Resolve any failing tests. The plugin MUST achieve ~90% functional logic coverage before WP10 is signed off.

## Activity Log

- 2026-02-27T23:18:47Z – Antigravity – shell_pid=55083 – lane=doing – Assigned agent via workflow command
- 2026-02-27T23:58:31Z – Antigravity – shell_pid=55083 – lane=for_review – Moved to for_review
- 2026-02-28T00:14:43Z – Antigravity – shell_pid=55083 – lane=done – Review passed: integration tests + separation of concerns verified

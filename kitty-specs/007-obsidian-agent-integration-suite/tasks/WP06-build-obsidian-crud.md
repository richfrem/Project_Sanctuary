---
work_package_id: WP06
title: Build Obsidian Vault CRUD Skill
lane: "done"
dependencies: []
base_branch: main
base_commit: 465d8e7f4bfa0673ecfdbc47b492271cbeeeae1e
created_at: '2026-02-27T22:24:54.088132+00:00'
subtasks: [T025, T026, T027, T028, T029]
shell_pid: "55083"
agent: "Antigravity"
reviewed_by: "richfrem"
review_status: "approved"
---

# Work Package Prompt: WP06 – Build Obsidian Vault CRUD Skill

## Objectives & Success Criteria
- Implement the baseline agent capability for reading, creating, and updating standard local Obsidian `.md` notes.
- Integrate POSIX atomic renames to prevent corruption when Obsidian auto-saves concurrently.

## Context & Constraints
- Must align strictly with the implementation pattern decided upon in the WP01 ADR.
- The `obsidian-markdown-mastery` utility handles syntax; WP06 strictly handles disk I/O and locking.

## Subtasks & Detailed Guidance

### Subtask T025 – Scaffold Plugin Framework
- **Purpose**: Prepare architecture.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-vault-crud/SKILL.md` and `scripts/`.

### Subtask T026 – Implement Atomic Writes
- **Purpose**: Prevent partial disk writes from crashing active clients.
- **Steps**: Any Python write mechanism MUST write the mutated data to a hidden `.tmp` file in the same directory, then perform `os.rename()` (which is atomic on POSIX) to instantly swap the old note for the new one.

### Subtask T027 – Implement `.agent-lock` protocol
- **Purpose**: Human-Active Vault Protection.
- **Steps**: Build logic that creates a bidirectional advisory `.agent-lock` file at the root of the vault before any write batch and removes it after. This does not strictly stop Obsidian, but governs agent-vs-agent. Optionally, add process-level detection (`pgrep` or equivalent checking for `.obsidian/workspace.json` lock) as a "warm vault" warning signal, not a hard gate.

### Subtask T028 – Detect Concurrent Edits
- **Purpose**: Avoid overwriting human inputs.
- **Steps**: Capture file `mtime` before reading. Before writing the `.tmp` file back over it, check `mtime` again. If it shifted, a user edited the file mid-agent-thought. Abort and alert.

### Subtask T029 – Lossless YAML Parsing
- **Purpose**: Prevent breaking Dataview.
- **Steps**: Ensure PyYAML is NOT used. Use `ruamel.yaml` to read/write the note frontmatter perfectly preserving comments, indentation, and array styles.

## Activity Log

- 2026-02-27T22:24:54Z – Antigravity – shell_pid=55083 – lane=doing – Assigned agent via workflow command
- 2026-02-27T22:43:34Z – Antigravity – shell_pid=55083 – lane=for_review – CRUD skill complete with atomic writes, locking, mtime guard, ruamel.yaml. Added obsidian-init onboarding skill with prereq installation docs. Bundled research resources for portability.
- 2026-02-28T00:10:52Z – Antigravity – shell_pid=55083 – lane=done – Review passed: verified git diff, code quality, and separation of concerns

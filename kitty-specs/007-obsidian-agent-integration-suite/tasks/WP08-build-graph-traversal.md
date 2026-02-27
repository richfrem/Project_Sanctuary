---
work_package_id: WP08
title: Build Obsidian Graph Traversal Skill
lane: "doing"
dependencies: []
base_branch: main
base_commit: 6cd6db93806a492d7c315ef22f06cbdf4b3d26be
created_at: '2026-02-27T22:59:16.685984+00:00'
subtasks: [T035, T036, T037, T038, T039]
shell_pid: "55083"
---

# Work Package Prompt: WP08 – Build Obsidian Graph Traversal Skill

## Objectives & Success Criteria
- Allow agents to query "What connects to Note X?" and receive an instant back/forward graph map.
- Performance < 2 seconds for deep queries without re-scanning the entire vault.

## Context & Constraints
- Needs to parse thousands of internal Markdown files rapidly.
- Relies on the `obsidian-parser` built in WP05 to find the links.

## Subtasks & Detailed Guidance

### Subtask T035 – Scaffold Plugin Framework
- **Purpose**: Establish root.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-graph-traversal/`. Add `SKILL.md` detailing the semantic bridging capabilities.

### Subtask T036 – Hook into `obsidian-parser`
- **Purpose**: Don't reinvent the wheel.
- **Steps**: Import the link extraction regex/AST logic from WP05. Ensure it safely filters out Image transclusions (`![[image.png]]`) while collecting note nodes.

### Subtask T037 – Build In-Memory Graph Index
- **Purpose**: Solve performance limits via cache.
- **Steps**: To avoid full-vault scans on every query, build a lightweight JSON or SQLite graph index mapping `Source -> Target`. Add an invalidation mechanism responding to file `mtime` changes.

### Subtask T038 – Forward/Backward Resolution
- **Purpose**: Graph logic endpoints.
- **Steps**: Write query logic traversing the index. E.g. `get_backlinks("Project Plan")` or `get_2nd_degree_connections("Concept A")`. 

### Subtask T039 – Synthetic Performance Testing
- **Purpose**: Verification of the 2-second target.
- **Steps**: Generate a synthetic graph of 50 interlinked test nodes. Write a python test that asserts the index generation and querying executes within boundary constraints.

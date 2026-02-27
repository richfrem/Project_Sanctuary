---
work_package_id: WP05
title: Build Obsidian Markdown Mastery Skill
lane: "doing"
dependencies: []
base_branch: main
base_commit: 0076d81527e4c8ac68076ac5fc46badb43271d1c
created_at: '2026-02-27T22:06:53.251039+00:00'
subtasks: [T020, T021, T022, T023, T024]
shell_pid: "55083"
agent: "Antigravity"
---

# Work Package Prompt: WP05 – Build Obsidian Markdown Mastery Skill

## Objectives & Success Criteria
- Implement the baseline parser required to format and read Obsidian distinct syntax correctly.
- Establish the `obsidian-parser` as a standalone utility module importable by other skills (like Graph Traversal).

## Context & Constraints
- Must map heading anchors, block references, wikilink aliases, and image transclusions distinctly.

## Subtasks & Detailed Guidance

### Subtask T020 – Scaffold Skill framework
- **Purpose**: Setup plugin root and shared configuration.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-markdown-mastery/SKILL.md` and the `scripts/` folder. Define a `plugins/obsidian-integration/config.yaml` or environment variable (`SANCTUARY_VAULT_PATH`) to ensure all downstream skills import from this single source.

### Subtask T021 – Build `obsidian-parser` Shared Utility
- **Purpose**: Prevent regex drift.
- **Steps**: Write a centralized Python module in `plugins/obsidian-integration/obsidian-parser/`. This is the universal gatekeeper for extracting and injecting wikilinks. Incorporate any applicable non-violating patterns from WP02 Kepano analysis into the shared `obsidian-parser`.

### Subtask T022 – Handle Edge Case Link Mappings
- **Purpose**: Account for Obsidian's proprietary capabilities identified by Red Team.
- **Steps**: Inside the parser, implement regex/AST logic specific to:
    - `[[Note#Heading]]` (Heading-level linking)
    - `[[Note#^block-id]]` (Block-level linking)
    - `[[Note|Alias Text]]` (Aliasing)
    - `![[Note]]` (Transclusion/Embed - Must be identified uniquely so the Graph Traverser doesn't parse it as a standard node).

### Subtask T023 – Callout Parsing Logic
- **Purpose**: Correct formatting of highlighted contexts.
- **Steps**: Add functions to wrap string sections in Obsidian-flavored Callouts (`> [!info]`). 

### Subtask T024 – Write Verification Tests
- **Purpose**: Prove resilience.
- **Steps**: Expose the parser to strings containing complex links and callouts. Assert that the correct types (Embed vs Link) and metadata (Alias, Heading, Block) are accurately extracted.

## Activity Log

- 2026-02-27T22:06:54Z – Antigravity – shell_pid=55083 – lane=doing – Assigned agent via workflow command

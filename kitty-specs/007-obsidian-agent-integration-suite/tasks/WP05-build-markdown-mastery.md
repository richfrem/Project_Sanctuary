---
work_package_id: "WP05"
title: "Build Obsidian Markdown Mastery Skill"
lane: "planned"
dependencies: ["WP01", "WP02"]
subtasks: ["T020", "T021", "T022", "T023", "T024"]
---

# Work Package Prompt: WP05 – Build Obsidian Markdown Mastery Skill

## Objectives & Success Criteria
- Implement the baseline parser required to format and read Obsidian distinct syntax correctly.
- Establish the `obsidian-parser` as a standalone utility module importable by other skills (like Graph Traversal).

## Context & Constraints
- Must map heading anchors, block references, wikilink aliases, and image transclusions distinctly.

## Subtasks & Detailed Guidance

### Subtask T020 – Scaffold Skill framework
- **Purpose**: Setup plugin root.
- **Steps**: Create `plugins/obsidian-integration/skills/obsidian-markdown-mastery/SKILL.md` and the `scripts/` folder.

### Subtask T021 – Build `obsidian-parser` Shared Utility
- **Purpose**: Prevent regex drift.
- **Steps**: Write a centralized Python module in `plugins/obsidian-integration/obsidian-parser/`. This is the universal gatekeeper for extracting and injecting wikilinks.

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

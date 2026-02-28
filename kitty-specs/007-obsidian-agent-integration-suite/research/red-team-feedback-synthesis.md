# Red Team Feedback Synthesis: Obsidian Integration Suite

**Date**: 2026-02-26
**Phase**: Phase 0.5 - Red Team Review
**Reviewers**: Grok, GPT-5.2, Gemini 3.1 Pro (Claude 4.6 Opus pending)

## 1. Executive Summary

The Red Team reviewers (Grok, GPT-5.2, Gemini 3.1 Pro, and Claude Opus 4.6) reached a consensus: the architectural decision to use a direct filesystem + multi-root workspace approach is sound and preserves tool sovereignty. However, the design assumes a "friendly" file state that Obsidian does not guarantee. 

Without explicit execution guardrails, the current architecture presents **Critical risks of silent vault corruption** and **inconsistent state exports**. Furthermore, execution tracking holds **Critical bugs** involving subtask ID collisions and missing dependency fields. 

## 2. Key Findings & Required Mitigations

### Finding 1: Concurrent Write Corruption (Critical)
*   **Risk**: Obsidian lacks OS-level file locking. Using `pathlib` to mutate `.md` or `.base` files while the user has the Obsidian app open risks race conditions, leading to missing edits, truncated frontmatter, or silent JSON corruption in `.canvas` files.
*   **Mitigation (Apply to WP03)**:
    1.  **Atomic Writes**: Agents must write to `.tmp` files and perform a POSIX atomic rename/move.
    2.  **Advisory Locking**: Implement an `.obsidian/.agent-lock` protocol or use `portalocker`. Refuse writes if the file `mtime` changed since read.
    3.  **Human-Active Detection**: Check `.obsidian/workspace.json` mtime. If the vault is actively being used by a human, the agent should downgrade to append-only or read-only mode, or warn the user.

### Finding 2: Inconsistent State Export (Critical)
*   **Risk**: WP05 (Forge Soul) aggregates sealed notes. If WP03 (CRUD) or WP06 (Legacy Scrubbing) runs concurrently, WP05 might export broken links or partially refactored notes into the Hugging Face JSONL.
*   **Mitigation (Apply to WP05)**:
    1.  **Transactional Export**: WP05 must enforce snapshot isolation. Take a tree hash of the vault before export.
    2.  **Fail-Fast `mtime` Check**: If any file's `mtime` changes during the export run, abort the export immediately to prevent poisoning the HF dataset.

### Finding 3: Parser Depth & Wikilink Ambiguity (High)
*   **Risk**: Obsidian Markdown is highly proprietary. Naive regex for wikilinks will break on aliases (`[[Note|Alias]]`), block references (`[[#^block]]`), embedded callouts, and Dataview arrays. Resolving `[[Note]]` requires a global index, otherwise forward-link resolution (WP04) will hit O(n) performance limits or hallucinate edges.
*   **Mitigation (Apply to WP04 / Shared)**:
    1.  **Shared Parser Module**: Create `plugins/obsidian-integration/obsidian-parser/` to handle all edge cases centrally instead of scattering regex across WP03/04/05/06.
    2.  **In-Memory Index**: WP04 needs to build a lightweight link resolver index (path + aliases) to avoid O(n) disk reads on every traversal.

### Finding 4: YAML Frontmatter Mutation Hazards (High)
*   **Risk**: Standard Python `frontmatter` libraries often reorder keys or normalize scalars unpredictably, which can break downstream Obsidian community plugins like Dataview that rely on type stability.
*   **Mitigation (Apply to WP03)**: 
    1.  **Lossless Round-Trip**: Enforce lossless YAML parsing (e.g., using `ruamel.yaml`) or reject the mutation if the format cannot be preserved accurately.

### Finding 5: Subtask ID Collision (Critical)
*   **Risk**: WP01 defines T005 as "Obtain human steward approval". WP02 defines T005 as "Analyze HF schema". This ID collision will break automated task tracking systems.
*   **Mitigation (Apply to WP02)**: Shift the T005 range in WP02 (and all subsequent WPs) forward so every subtask ID is globally unique.

### Finding 6: Empty Dependency Fields (High)
*   **Risk**: Every Work Package has `dependencies: []` in its frontmatter, despite `plan.md` clearly defining a sequential execution order.
*   **Mitigation (Apply to all WPs)**: Populate the YAML frontmatter `dependencies` arrays to correctly map the execution flow defined in `plan.md`.

### Finding 7: Orphaned Skill Breakdown (High)
*   **Risk**: `obsidian-plugin-architecture.md` lists 6 skills. WP04 groups some together, but `obsidian-markdown-mastery` is entirely orphaned without a dedicated implementation task.
*   **Mitigation (Apply to WPs/Plan)**: Expand the Work Packages. Add `WP07-build-markdown-mastery-skill.md` and `WP08-build-dynamic-views-skill.md` (for `.base` and `.canvas`) to ensure every architecture pillar has an execution path.

### Finding 8: Execution Feasibility & Reordering (Low)
*   **Risk**: WP06 runs after WP03-05. Refactoring links *after* testing graph traversal makes testing harder.
*   **Mitigation (Apply to WPs/Plan)**:
    1.  Move WP06 to execute *before* WP03/WP04, so the graph is clean before traversal tests.
    2.  Mandate a "Dry-Run" mode for WP06 to prevent aggressive regex from destroying the vault.

## 3. Action Plan

1.  Update `plan.md` to explicitly list the 6 skills and spread them across explicitly defined Work Packages.
2.  Fix Subtask ID Collisions (Renumber WP02-WP06 sequentially).
3.  Populate `dependencies: [...]` arrays in every WP frontmatter.
4.  Refactor WP03 to demand Atomic Writes, Lossless YAML, and Human-Active detection.
5.  Refactor WP04/05 to demand the shared parser, index cache, and snapshot isolation.
6.  Create new WP prompt files for the orphaned skills (Markdown Mastery, Dynamic Views, Integration Testing).

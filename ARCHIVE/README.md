# ARCHIVE

Preserved material from before the move to the APM plugin architecture. Nothing here is in force; it is kept for lineage.

## Why `git add -f`

`.gitignore` ignores `ARCHIVE/` (and any `archive/` directory), so a plain `git add` silently skips new files here. Tracked files stay
tracked, but anything new must be added with `git add -f <path>`. (The negation `!ARCHIVE/council_orchestrator/` in `.gitignore` has no
effect because git cannot re-include a path whose parent directory is ignored.)

## What is tracked, and what is deliberately not

Tracked (force-added): human-authored documents and small authored sources, for example `legacy_protocols/` (protocols deprecated 2026-11-03),
`ADRs/`, `forge/`, `legacy_agent_learning/`, and (added 2026-11-03) the blueprints, policy drafts, council-orchestrator docs, the
`recursive-tool-smithing` plugin, `mnemonic_cortex/` docs, awakening seeds under `dataset_package/`, and planning notes.

Not tracked on purpose (derived, generated, third-party or runtime data; recoverable from git history or regenerable):
- the retired ChromaDB vector store (`legacy_tools/data/`, `**/.vector_data/`): a derived copy of the source documents
- genome snapshots in `dataset_package/` (`markdown_snapshot_full_genome_llm_distilled.txt`, `sanctuary_whole_genome_data.jsonl`): generated from the source documents
- caches and dependencies (`__pycache__`, `.pytest_cache`, `node_modules`, `.obsidian`, `.DS_Store`), model files, `skills-lock.json`
- council runtime directives (`07_COUNCIL_AGENTS/directives/`), generated context bundles, third-party research PDFs

The old `tools/`, `mcp_servers/`, `00_CHRONICLE/` and similar directories are in the repository history; the last commit before the pivot is
`a2952046a` (2026-03-12).

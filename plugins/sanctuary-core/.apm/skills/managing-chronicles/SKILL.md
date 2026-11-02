---
name: managing-chronicles
description: "Manages the Living Chronicle project journal. Creates, lists, searches, and views historical chronicle entries across the project repository. Use when documenting project milestones, journaling session context, or searching historical decisions."
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Managing Chronicles (`managing-chronicles`)

Manages the Living Chronicle (`chronicle/entries/`) — Project Sanctuary's historical journal of decisions, milestones, architectural evolutions, and operational reflections.

## Contents
- [Constraints](#constraints) • [Quick Start](#quick-start) • [Workflow](#workflow) • [Verification](#verification) • [References](#references)

## Constraints

1. Never overwrite existing historical entries; append or transition status via sequential IDs.
2. Chronicles reside under `chronicle/entries/` with automated RLM indexing in `chronicle/ledger.json`.

## Quick Start

```bash
# Create an entry
python3 scripts/chronicle_manager.py create "Title" --content "Details..." --author "Agent"

# List recent entries
python3 scripts/chronicle_manager.py list --limit 5

# View specific entry
python3 scripts/chronicle_manager.py get 342

# Search chronicles
python3 scripts/chronicle_manager.py search "keyword"
```

## Workflow

1. **Verify Existing Records**: Run `search` or `list` to check whether a topic or decision has already been journaled.
2. **Author Entry**: Draft the entry with clear context, title, author, and status (`draft`, `published`, or `canonical`).
3. **Persist**: Execute `chronicle_manager.py create` to generate the next numbered markdown chronicle file in `chronicle/entries/`.

## Verification

```bash
python3 scripts/chronicle_manager.py list --limit 1
```

## References
- Schema & Lifecycle: [chronicle-schema.md](references/chronicle-schema.md)

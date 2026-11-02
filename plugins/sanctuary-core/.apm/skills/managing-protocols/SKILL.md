---
name: managing-protocols
description: "Manages the Project Sanctuary governance protocols in protocols/. Creates, lists, searches, and transitions protocol lifecycle states (PROPOSED, CANONICAL, DEPRECATED). Use when defining architectural standards, ratifying procedures, or reviewing governance frameworks."
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Managing Protocols (`managing-protocols`)

Manages Project Sanctuary's formal governance framework and operating protocols located in `protocols/`.

## Contents
- [Constraints](#constraints) • [Quick Start](#quick-start) • [Workflow](#workflow) • [Verification](#verification) • [References](#references)

## Constraints
1. New protocols must be created with `PROPOSED` status.
2. Status transitions to `CANONICAL` or `DEPRECATED` require an explicit reason note.
3. Protocols reside under `protocols/`.

## Quick Start

```bash
# Create a protocol
python3 scripts/protocol_manager.py create "Title" --content "Details..." --author "Agent"

# List canonical protocols
python3 scripts/protocol_manager.py list --status CANONICAL

# View protocol
python3 scripts/protocol_manager.py get 128

# Search protocols
python3 scripts/protocol_manager.py search "Protocol 128"
```

## Workflow

1. **Check Existing Frameworks**: Search `protocols/` to verify if a protocol exists.
2. **Draft Protocol**: Run `protocol_manager.py create` with title, author, and content.
3. **Ratification / Lifecycle Update**: Once tested and ratified, update status to `CANONICAL` via `protocol_manager.py update`.

## Verification

```bash
python3 scripts/protocol_manager.py list --limit 1
```

## References
- Schema & Lifecycle: [protocol-schema.md](references/protocol-schema.md)

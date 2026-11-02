# Protocol Schema & Governance Lifecycle

Protocol documents (`protocols/`) establish Project Sanctuary's formal governance, architectural rules, operational standards, and safety boundaries.

## CLI Usage

Run commands from the repository root:

```bash
# Create a new protocol
python3 plugins/sanctuary-core/.apm/skills/managing-protocols/scripts/protocol_manager.py create "Title" --content "Details" [--author "Agent"]

# List protocols
python3 plugins/sanctuary-core/.apm/skills/managing-protocols/scripts/protocol_manager.py list [--limit 10] [--status PROPOSED|CANONICAL|DEPRECATED]

# View a specific protocol
python3 plugins/sanctuary-core/.apm/skills/managing-protocols/scripts/protocol_manager.py get <number>

# Search protocols
python3 plugins/sanctuary-core/.apm/skills/managing-protocols/scripts/protocol_manager.py search "keyword"

# Transition protocol status
python3 plugins/sanctuary-core/.apm/skills/managing-protocols/scripts/protocol_manager.py update <number> --status CANONICAL --reason "Ratified after review"
```

## Governance Lifecycle
- `PROPOSED`: Draft protocol awaiting review.
- `CANONICAL`: Ratified, active governance protocol (e.g. Protocol 128).
- `DEPRECATED`: Superseded historical protocol.

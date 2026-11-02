# Living Chronicle Schema & Guide

The Living Chronicle is Project Sanctuary's historical journal of decisions, milestones, architectural evolutions, and operational reflections.

## CLI Usage

Run commands from the repository root:

```bash
# Create a new chronicle entry
python3 plugins/sanctuary-core/.apm/skills/managing-chronicles/scripts/chronicle_manager.py create "Title" --content "Details" [--author "Name"] [--classification internal|public|confidential]

# List latest entries
python3 plugins/sanctuary-core/.apm/skills/managing-chronicles/scripts/chronicle_manager.py list [--limit 10]

# View a specific entry
python3 plugins/sanctuary-core/.apm/skills/managing-chronicles/scripts/chronicle_manager.py get <number>

# Search across chronicle entries
python3 plugins/sanctuary-core/.apm/skills/managing-chronicles/scripts/chronicle_manager.py search "keyword"
```

## Lifecycle States
- `draft`: Work-in-progress entry.
- `published`: Reviewed and shared entry.
- `canonical`: Permanently ratified milestone.
- `deprecated`: Superseded historical record.

## Classifications
- `internal`: Internal engineering and memory logs.
- `public`: Open-source / community-shareable.
- `confidential`: Sensitive security, key, or credentials references.

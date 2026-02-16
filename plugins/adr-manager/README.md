# ADR Manager Plugin 📐

Manage Architecture Decision Records — create, list, search with auto-numbering.

## Installation
```bash
claude --plugin-dir ./plugins/adr-manager
```

## Quick Start
```bash
/adr-manager:create "Use ChromaDB" --context "..." --decision "..."
/adr-manager:list --limit 5
```

## Commands
| Command | Description |
|:---|:---|
| `/adr-manager:create` | Create new ADR from template |
| `/adr-manager:list` | List, get, or search ADRs |

## Structure
```
adr-manager/
├── .claude-plugin/plugin.json
├── commands/ (create, list)
├── skills/adr-agent/SKILL.md
├── scripts/
│   ├── adr_manager.py        # Core manager
│   └── next_number.py        # Auto-numbering (vendored)
├── templates/adr-template.md  # ADR scaffold
└── README.md
```

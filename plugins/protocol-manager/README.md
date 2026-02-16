# Protocol Manager Plugin ⚖️

Manage Protocol documents — create, list, search, update with auto-numbering and status tracking.

## Installation
```bash
claude --plugin-dir ./plugins/protocol-manager
```

## Commands
| Command | Description |
|:---|:---|
| `/protocol-manager:create` | Create new protocol with auto-numbering |
| `/protocol-manager:manage` | List, view, search, or update protocols |

## Status Lifecycle
🟡 PROPOSED → 🟢 CANONICAL → 🔴 DEPRECATED

## Structure
```
protocol-manager/
├── .claude-plugin/plugin.json
├── commands/ (create, manage)
├── skills/protocol-agent/SKILL.md
├── scripts/protocol_manager.py   # Standalone (zero deps)
└── README.md
```

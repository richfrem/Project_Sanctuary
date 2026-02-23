# Chronicle Manager Plugin 📜

Living Chronicle journaling — manage project event entries with status and classification.

## Installation
```bash
claude --plugin-dir ./plugins/chronicle-manager
```

## Commands
| Command | Description |
|:---|:---|
| `/chronicle-manager:create` | Create new chronicle entry |
| `/chronicle-manager:manage` | List, view, or search entries |

## Status Lifecycle
📝 draft → 📗 published → 🏛️ canonical → 🔴 deprecated

## Structure
```
chronicle-manager/
├── .claude-plugin/plugin.json
├── commands/ (create, manage)
├── skills/chronicle-agent/SKILL.md
├── scripts/chronicle_manager.py   # Standalone (zero deps)
└── README.md
```

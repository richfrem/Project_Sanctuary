# Task Manager Plugin 📋

Lightweight kanban task board — JSON-backed with zero dependencies.

## Installation
```bash
claude --plugin-dir ./plugins/task-manager
```

## Quick Start
```bash
/task-manager:create "Fix login bug" --objective "Resolve 401 errors"
/task-manager:board
/task-manager:move 1 in-progress
/task-manager:list --status done
```

## Commands

| Command | Description |
|:---|:---|
| `/task-manager:create` | Create a new task |
| `/task-manager:list` | List / filter tasks |
| `/task-manager:move` | Move task between lanes |
| `/task-manager:board` | Show kanban board |

## Lanes
`backlog` → `todo` → `in-progress` → `done`

## Data
Tasks stored at `tasks/tasks.json` (auto-created). Override with `--file path`.

## Structure
```
task-manager/
├── .claude-plugin/plugin.json
├── commands/ (create, list, move, board)
├── skills/task-agent/SKILL.md
├── scripts/task_manager.py
├── docs/task-manager-workflow.mmd
└── README.md
```

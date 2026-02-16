# Dependency Management Plugin 💊

Python dependency management with pip-compile locked-file workflow for the MCP server fleet.

## Installation
```bash
claude --plugin-dir ./plugins/dependency-management
```

## Commands
| Command | Description |
|:---|:---|
| `/dependency-management:manage` | Add, upgrade, or security-patch a dependency |
| `/dependency-management:audit` | Audit tree for conflicts, stale pins, compliance |

## Core Rules
1. No manual `pip install` — use `.in` → `pip-compile` → `.txt`
2. Commit `.in` + `.txt` together
3. Core → Service-specific → Dev-only tiered hierarchy
4. Dockerfiles: only `COPY` + `pip install -r`

## Structure
```
dependency-management/
├── .claude-plugin/plugin.json
├── commands/ (manage, audit)
├── skills/dependency-agent/SKILL.md
└── README.md
```

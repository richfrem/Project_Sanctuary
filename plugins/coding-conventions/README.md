# Coding Conventions Plugin 📝

Coding standards and header templates for Python, TypeScript/JavaScript, and C#/.NET.

## Installation
```bash
claude --plugin-dir ./plugins/coding-conventions
```

## Commands
| Command | Description |
|:---|:---|
| `/coding-conventions:apply` | Review code or generate compliant headers |

## Templates Included
- `templates/python-tool-header-template.py` — Extended Python header
- `templates/js-tool-header-template.js` — JS/TS header

## Structure
```
coding-conventions/
├── .claude-plugin/plugin.json
├── commands/apply.md
├── skills/conventions-agent/SKILL.md
├── templates/
│   ├── python-tool-header-template.py
│   └── js-tool-header-template.js
└── README.md
```

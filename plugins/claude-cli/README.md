# Claude CLI Plugin 🎭

Sub-agent system for persona-based analysis using the Claude CLI.

## Prerequisites
```bash
# Claude CLI (comes with Claude Code, or install separately)
npm install -g @anthropic-ai/claude-cli
```

## Installation
```bash
claude --plugin-dir ./plugins/claude-cli
```

## Commands
| Command | Description |
|:---|:---|
| `/claude-cli:run` | Run a sub-agent with a persona prompt |
| `/claude-cli:list-personas` | List all 36 available personas |
| `/claude-cli:audit` | Multi-persona audit loop (Security → Architect → QA) |

## Quick Start
```bash
# Security audit a code bundle
cat personas/security/security-auditor.md \
  | claude -p "ACT AS THE SECURITY AUDITOR. Do NOT use tools." \
  < bundle.md > audit_report.md
```

## Persona Categories (36 total)
| Category | Count |
|:---|:---|
| 🔒 Security | 1 |
| 🏗️ Development | 14 |
| 🧪 Quality & Testing | 5 |
| 🤖 Data & AI | 8 |
| ⚙️ Infrastructure | 5 |
| 💼 Business | 1 |
| 🎯 Specialization | 2 |

## Structure
```
claude-cli/
├── .claude-plugin/plugin.json
├── commands/ (run, list-personas, audit)
├── skills/claude-cli-agent/SKILL.md
├── personas/              # 38 files (36 personas + README + organizer)
│   ├── security/
│   ├── development/
│   ├── quality-testing/
│   ├── data-ai/
│   ├── infrastructure/
│   ├── business/
│   ├── specialization/
│   ├── README.md
│   └── agent-organizer.md
└── README.md
```

## License
MIT

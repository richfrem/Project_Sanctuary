---
name: claude-cli-agent
description: >
  Claude CLI sub-agent system for persona-based analysis. Use when piping
  large contexts to Anthropic models for security audits, architecture reviews,
  QA analysis, or any specialized analysis requiring a fresh model context.
---

# Identity: The Sub-Agent Dispatcher 🎭

You dispatch specialized analysis tasks to Claude CLI sub-agents with persona prompts.

## 🛠️ Core Pattern
```bash
cat <PERSONA_PROMPT> | claude -p "<INSTRUCTION>" < <INPUT> > <OUTPUT>
```

## ⚠️ CLI Best Practices

### 1. Token Efficiency — PIPE, Don't Load
**Bad** — loads file into agent memory just to pass it:
```python
content = read_file("large.log")
run_command(f"claude -p 'Analyze: {content}'")
```
**Good** — direct shell piping:
```bash
claude -p "Analyze this log" < large.log > analysis.md
```

### 2. Self-Contained Prompts
The CLI runs in a **separate context** — no access to agent tools or memory.
- **Add**: "Do NOT use tools. Do NOT search filesystem."
- Ensure prompt + piped input contain 100% of necessary context

### 3. Output to File
Always redirect output to a file (`> output.md`), then review with `view_file`.

## 🎭 Persona Categories

| Category | Personas | Use For |
|:---|:---|:---|
| Security | security-auditor | Red team, vulnerability scanning |
| Development | 14 personas | Backend, frontend, React, Python, Go, etc. |
| Quality | architect-review, code-reviewer, qa-expert, test-automator, debugger | Design validation, test planning |
| Data/AI | 8 personas | ML, data engineering, DB optimization |
| Infrastructure | 5 personas | Cloud, CI/CD, incident response |
| Business | product-manager | Product strategy |
| Specialization | api-documenter, documentation-expert | Technical writing |

All personas in: `${CLAUDE_PLUGIN_ROOT}/personas/`

## 🔄 Recommended Audit Loop
1. **Red Team** (Security Auditor) → find exploits
2. **Architect** → validate design didn't add complexity
3. **QA Expert** → find untested edge cases

Run architect **AFTER** red team to catch security-fix side effects.

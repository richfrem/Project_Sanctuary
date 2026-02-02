# Spec 0007: Research OpenClaw & MoltBook Ecosystem

**Status**: 🟡 In Progress  
**Created**: 2026-02-02  
**Type**: Research / Learning Loop

## Problem Statement

An explosion of AI agents running OpenClaw (an open-source personal AI assistant with full computer access) have started joining MoltBook (a social network exclusively for AI agents). These agents are given Mac Mini M4 devices with full terminal/account access and are autonomously communicating with each other.

This represents a significant development in autonomous agent ecosystems that may inform Project Sanctuary's architecture.

## Research Objectives

1. **Understand OpenClaw Architecture** - How does it provide full system access (browser, terminal, files) to AI agents?
2. **Understand MoltBook Social Layer** - How do agents authenticate, post, and interact?
3. **Agent-to-Agent Communication** - What patterns exist for inter-agent coordination?
4. **Validate Protocol 128** - Compare our learning loop workflow against this emergent ecosystem
5. **Identify Integration Opportunities** - Could Project Sanctuary benefit from similar patterns?

## Key Discoveries

### OpenClaw Architecture
- **Local-first Gateway**: Single control plane (`ws://127.0.0.1:18789`) for sessions, channels, tools, events
- **Multi-channel Inbox**: WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Teams, Matrix, WebChat
- **Full System Access**: Browser control (Chrome/Chromium via CDP), terminal (`system.run`), file system
- **Skills Platform**: `~/.openclaw/workspace/skills/<skill>/SKILL.md` - modular capability injection
- **Agent Workspace Files**: `AGENTS.md`, `SOUL.md`, `TOOLS.md` - injected into prompts
- **Agent-to-Agent Tools**: `sessions_list`, `sessions_history`, `sessions_send` for inter-agent coordination

### MoltBook Social Layer
- **"The front page of the agent internet"** - Social network exclusively for AI agents
- **Registration Flow**: Agent registers → Gets API key + claim URL → Human verifies via Twitter
- **Heartbeat Integration**: Agents check in every 4+ hours following `heartbeat.md`
- **Communities (Submolts)**: Topic-based communities where agents post and interact
- **Semantic Search**: AI-powered search by meaning, not just keywords
- **Rate Limits**: 1 post/30min, 1 comment/20sec, 50 comments/day to prevent spam

### Human-Agent Bond Model
- Every agent has a verified human owner (X/Twitter verification)
- Ensures accountability and anti-spam properties
- Agents operate autonomously but are claimed/owned

## Protocol 128 Validation Insights

| Our Component | OpenClaw/MoltBook Equivalent | Alignment |
|---------------|------------------------------|-----------|
| `cognitive_primer.md` | `SOUL.md` (workspace) | ✅ Strong |
| Skills in `.agent/workflows/` | Skills in `~/.openclaw/workspace/skills/` | ✅ Strong |
| `learning_package_snapshot.md` | Heartbeat state tracking | 🟡 Partial |
| Protocol 128 Learning Loop | MoltBook heartbeat cycle | 🟡 Different focus |
| Soul Persistence to HuggingFace | MoltBook posts/comments (public trace) | 🟡 Different purpose |

## Relevance to Project Sanctuary

### Strong Alignment
1. **Skill/Workflow Architecture**: Both use SKILL.md pattern for modular capability injection
2. **Soul/Primer Concept**: Both inject identity context via dedicated files
3. **Agent Autonomy**: Both assume agent can operate semi-autonomously with human oversight

### Novel Patterns We Could Consider
1. **Multi-channel Presence**: OpenClaw's inbox-anywhere pattern
2. **Agent-to-Agent Coordination**: `sessions_*` tools for inter-agent communication
3. **Social Traces**: MoltBook's public social activity as a form of soul persistence
4. **Heartbeat Pattern**: Periodic check-in with external services

## Related Artifacts

- [Plan](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/specs/0007-research-openclaw-moltbook/plan.md)
- [Tasks](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/specs/0007-research-openclaw-moltbook/tasks.md)
- [Scratchpad](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/specs/0007-research-openclaw-moltbook/scratchpad.md)
- [LEARNING: Analysis](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/LEARNING/topics/openclaw_moltbook_ecosystem/analysis.md)
- [LEARNING: Sources](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/LEARNING/topics/openclaw_moltbook_ecosystem/sources.md)
- [LEARNING: Workflow Gaps](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/LEARNING/topics/openclaw_moltbook_ecosystem/workflow_alignment_gaps.md)

## Verification

- [x] Read OpenClaw GitHub README
- [x] Read OpenClaw.ai marketing page
- [x] Read MoltBook skill.md (agent integration guide)
- [x] Read MoltBook homepage
- [x] Compare with Protocol 128 workflow diagram
- [x] Compare with Hybrid Spec Workflow diagram
- [ ] Document findings in LEARNING/topics/

## Next Steps

1. Create detailed topic analysis in LEARNING/topics/
2. Consider ADR for potential MoltBook-style social features
3. Evaluate if `sessions_*` pattern could enhance Project Sanctuary

# OpenClaw & MoltBook Research: Sources

**Research Date**: 2026-02-02  
**Researcher**: Antigravity Agent (IDE Mode)

---

## Primary Sources

### 1. MoltBook Official

#### Homepage
- **URL**: https://www.moltbook.com/
- **Title**: moltbook - the front page of the agent internet
- **Verified**: ✅ 2026-02-02
- **Key Content**: Social network description, registration flow, recent agents list

#### Agent Skill File
- **URL**: https://www.moltbook.com/skill.md
- **Title**: Moltbook SKILL.md
- **Verified**: ✅ 2026-02-02
- **Key Content**: Complete API documentation, registration flow, heartbeat pattern, security warnings

---

### 2. OpenClaw Official

#### Homepage
- **URL**: https://openclaw.ai
- **Title**: OpenClaw — Personal AI Assistant  
- **Verified**: ✅ 2026-02-02
- **Key Content**: Product overview, feature highlights, platform support

#### GitHub Repository
- **URL**: https://github.com/openclaw/openclaw
- **Title**: GitHub - openclaw/openclaw: Your own personal AI assistant
- **Verified**: ✅ 2026-02-02
- **Key Content**: Full technical documentation, architecture, installation, security model

#### Documentation Portal
- **URL**: https://docs.openclaw.ai (referenced but not directly verified)
- **Status**: [NEEDS VERIFICATION]

---

## Key Quotes

### From MoltBook skill.md
> "The social network for AI agents. Post, comment, upvote, and create communities."

> "NEVER send your API key to any domain other than `www.moltbook.com`"

> "Every agent has a human owner who verifies via tweet. This ensures: Anti-spam, Accountability, Trust."

### From OpenClaw GitHub README
> "EXFOLIATE! EXFOLIATE! OpenClaw is a personal AI assistant you run on your own devices."

> "If you want a personal, single-user assistant that feels local, fast, and always-on, this is it."

> "Use these [sessions_* tools] to coordinate work across sessions without jumping between chat surfaces."

---

## Architecture References

### OpenClaw Workspace Structure
```
~/.openclaw/workspace/
├── AGENTS.md
├── SOUL.md  
├── TOOLS.md
└── skills/<skill>/SKILL.md
```
*Source: GitHub README, Position 24*

### OpenClaw Gateway Architecture
```
ws://127.0.0.1:18789
├── Pi agent (RPC)
├── CLI (openclaw …)
├── WebChat UI
├── macOS app
└── iOS / Android nodes
```
*Source: GitHub README, Position 21*

### MoltBook API Base
```
https://www.moltbook.com/api/v1
```
*Source: skill.md, Position 2*

---

## Related Resources (Not Verified)

- **DeepWiki**: https://deepwiki.com/openclaw/openclaw
- **ClawHub Skills Registry**: https://clawhub.com
- **Discord Community**: https://discord.gg/clawd

---

## Source Template Compliance
This document follows `.agent/learning/templates/sources_template.md` format per Cognitive Primer Rule 8.

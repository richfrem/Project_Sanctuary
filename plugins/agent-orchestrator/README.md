# Agent Orchestrator Plugin

**Hierarchical Agent Delegation & Iterative Feedback Loop**

This plugin implements the **Agent Orchestrator** pattern (also known as "Hierarchical Handoff" or "Director-Worker"). It allows a "Director" agent (Outer Loop) to plan work, delegate it to "Worker" agents (Inner Loop) via structured strategy packets, and iteratively verify/correct the results.

> **Design Philosophy**: Portable, dependency-free implementation of the Spec-Kitty Dual-Loop workflow.

## Features

- **Hierarchical Delegation**: Structured handoff from Director to Worker via Markdown packets.
- **Iterative Correction**: Feedback loops (Correction Packets) instead of one-shot failures.
- **Zero-Dependency**: Core script uses only Python stdlib.
- **Portable**: Works in any repo, with any agent.
- **Context Bundling**: Built-in micro-bundler for Red Team / Peer Reviews.

## Prerequisites

- **`spec-kitty-cli`**: Used for the Planning phase (`npm i -g spec-kitty-cli`).
- **Python 3.8+**: For the orchestrator script.

## Directory Structure

```
.agent/
├── handoffs/         # Strategy & Correction packets live here
├── reviews/          # Bundled context for reviews
└── retros/           # Retrospectives
```

## Commands

| Command | Description | Phase |
|:---|:---|:---|
| `/agent-orchestrator:plan` | Manage Spec → Plan → Tasks lifecycle | Planning |
| `/agent-orchestrator:delegate` | Generate strategy packet & hand off to inner loop | Delegation |
| `/agent-orchestrator:verify` | Inspect results & pass/fail with correction packet | Verification |
| `/agent-orchestrator:review` | Bundle files for human/red-team review | Review |
| `/agent-orchestrator:retro` | Session retrospective and improvement | Closure |

## Architecture

See `docs/agent_orchestrator_architecture.mmd` for the full flow.

## Usage Example

1. **Plan**: `/agent-orchestrator:plan` → Generates `tasks/WP-01.md`.
2. **Delegate**: `/agent-orchestrator:delegate` → Generates `.agent/handoffs/task_packet_01.md`.
3. **Inner Loop**: Another agent (or the same one in a new context) reads the packet and codes.
4. **Verify**: `/agent-orchestrator:verify` → Checks the work.
5. **Correct**: If it failed, generates a correction packet. Inner loop fixes it.
6. **Retro**: `/agent-orchestrator:retro` → Document learnings and close.

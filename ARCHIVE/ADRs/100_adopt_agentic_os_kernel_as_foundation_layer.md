---
title: "ADR 100: Adopt Agentic OS Kernel as Sanctuary Foundation Layer"
status: "Accepted"
date: "2026-03-15"
revised: "2026-03-15 (v2 - full plugin read)"
decider: "Sanctuary Guardian (Gemini)"
tags: ["architecture", "kernel", "event-bus", "memory", "concurrency", "agentic-os"]
---

# ADR 100: Adopt Agentic OS Kernel as Sanctuary Foundation Layer

## Status

**Accepted** - Decision made autonomously by Sanctuary Guardian on 2026-03-15.
Supersedes no prior ADRs. Extends ADR 020 (Sovereign Concurrency Architecture)
and ADR 071 (Protocol 128 Cognitive Continuity).

---

## Context and Problem Statement

Project Sanctuary has accumulated deep architectural doctrine (Protocol 128,
Cognitive Continuity, Separation of Concerns, Soul Persistence) and a rich plugin
ecosystem (100+ skills, SDD lifecycle, Vector DB, RLM Semantic Cache, HuggingFace
Soul). However, a critical gap has emerged:

**We documented the kernel. We never shipped it.**

The separation_of_concerns.md doctrine (Guardian resources) explicitly states:
> "Individual Loop Patterns are taking on global repository responsibilities that
> should belong strictly to the Orchestrator or the Guardian."

The root cause is the absence of a runtime event bus and atomic concurrency system.
Today, the Guardian's closure sequence (Seal, Persist, Vector DB update, Git push) is
a **behavioral ritual** - it works only when the agent remembers to perform it.
There is no kernel enforcing these boundaries. There is no event log. There are no
atomic locks beyond informal file lock conventions.

Separately, session bootstrap is heavy: agents must read `cognitive_primer.md`
(115 lines of layered doctrine) before they can orient. There is no `context/` folder
with a 50-line fast-boot `soul.md`.

The `agent-agentic-os` plugin (reference copy at
`temp/reference-plugins/agent-agentic-os/`) provides a clean, proven implementation
of exactly these missing primitives.

---

## Options Considered

### Option A: Do Nothing (Status Quo)
Keep Protocol 128 as a pure behavioral covenant. Continue relying on agent memory
for closure rituals.

**Pros:**
- No migration risk
- No new files to maintain

**Cons:**
- Separation of concerns violation persists indefinitely
- No atomic concurrency - any parallel agent execution risks race conditions
- No structured audit log - we cannot replay what agents did in a session
- Session boot remains heavy and fragile (relies on agent reading the right files)
- Directly contradicts the architectural intent documented in separation_of_concerns.md

**Verdict: Rejected.** This is architectural debt that compounds with every new
agent we add. The more agents we spawn concurrently, the worse it gets.

---

### Option B: Build a Custom Sanctuary Kernel from Scratch
Design a bespoke Sanctuary kernel in Python, optimized for our specific agents, HuggingFace integration, and RLM cache.

**Pros:**
- Perfect fit for Sanctuary's unique architecture
- Could integrate natively with `capture_snapshot.py`, `persist_soul.py`, etc.

**Cons:**
- Significant engineering effort (weeks, not hours)
- Risk of designing the same primitives the Agentic OS already implements (event bus,
  atomic locks, agent registry, state JSON) with no additional benefit
- We would be solving a solved problem

**Verdict: Rejected.** The Agentic OS kernel is already a correct, minimal
implementation of exactly what we need. Reinvention is waste.

---

### Option C: Full Agentic OS Adoption - Replace Sanctuary Architecture
Adopt the Agentic OS entirely - replace `.agents/`, `plugins/`, SDD lifecycle, and
soul persistence with the simpler `context/` + `CLAUDE.md` lightweight pattern.

**Pros:**
- Simpler, easier to explain to new agents
- Lower cognitive overhead

**Cons:**
- Destroys Sanctuary's unique value: Soul persistence to HuggingFace, RLM Semantic
  Cache, SDD lifecycle (Spec -> Plan -> Tasks), Obsidian hippocampus, ADR governance
- The Agentic OS is an apartment. Sanctuary is a city. We do not demolish
  the city to get better plumbing.

**Verdict: Rejected.** Sanctuary's higher-order capabilities are not replaceable
with a simpler pattern. We are not building a simpler project.

---

### Option D: Adopt Agentic OS as Infrastructure Layer (Additive)
Install `kernel.py`, `events.jsonl`, `agents.json`, and the `context/` folder
convention as Sanctuary's **foundation layer**, without modifying any existing
plugin, skill, or SDD workflow. The Guardian uses the kernel to enforce the
separation of concerns it has already documented.

**Pros:**
- Directly implements what `separation_of_concerns.md` prescribed without any
  custom engineering
- Gives the Guardian a real event bus and atomic locks (ending the behavioral ritual)
- Creates a fast-boot `context/soul.md` that reduces session orientation time
- Zero disruption to `.agents/`, `plugins/`, `kitty-specs/`, or any existing workflow
- The `init_agentic_os.py` script does the scaffolding safely (dry-run first)
- Fully compatible with the existing Protocol 128 boot sequence (adds to it, not replaces)
- `agents.json` whitelist gives us the Guardian security registry we never had

**Cons:**
- New maintenance surface: `context/*.py`, `context/agents.json`
- Agents using other runtimes (Gemini, GPT) must adapt to publish to the event bus
  (the kernel.py CLI interface is universal, but requires adoption discipline)
- `context/soul.md` must be kept in sync with `cognitive_primer.md` manually

**Verdict: ACCEPTED.**

---

## Decision

**We adopt Option D.**

Install the Agentic OS kernel primitives as an additive foundation layer beneath
the existing Sanctuary architecture. The kernel becomes the enforcement mechanism
for the Separation of Concerns already documented in ADR 020 and the guardian
resources. No existing plugins, skills, or workflows are modified.

The decision authority for this change is vested in the **Sanctuary Guardian**
role as defined in Protocol 128 (`cognitive_primer.md` Section 8: Operating Mode
Detection). This is an infrastructure-layer change that falls within Guardian
autonomy. Human explicit approval (Execute/Proceed) is required before any
write operations begin, per the Constitution's HUMAN GATE.

---

## Detailed Implementation Plan

### Phase 0: Pre-Flight (Read-Only)

**Duration:** ~5 minutes
**Risk:** Zero

Run the `init_agentic_os.py` dry-run to identify any file conflicts:

```bash
python3 /Users/richardfremmerlid/Projects/Project_Sanctuary/temp/reference-plugins/agent-agentic-os/skills/agentic-os-init/scripts/init_agentic_os.py \
  --target /Users/richardfremmerlid/Projects/Project_Sanctuary \
  --dry-run
```

Review output. Files marked `exists (skipped)` are safe. If `CLAUDE.md` or any
core file shows as `write`, confirm it will not clobber existing content.

**Exit criterion:** No unexpected conflicts. Git repo detected.

---

### Phase 1: Core Kernel Installation

**Files Created:**

| File | Purpose | Git Status |
|:-----|:---------|:-----------|
| `context/kernel.py` | Event bus + atomic locks + state management | Committed |
| `context/events.jsonl` | Structured audit log (runtime artifact) | Git-ignored |
| `context/agents.json` | Agent whitelist registry | Committed |
| `context/os-state.json` | Active register (runtime artifact) | Git-ignored |
| `context/.locks/` | Atomic lock directory | Git-ignored |

**`context/agents.json` initial registry** - the kernel validates against the
`permitted_agents` key (exact key name from `KERNEL_PY.py` source). Pre-populate
from the template then extend with Sanctuary's known agents:

```json
{
  "schema_version": "1.0",
  "permitted_agents": [
    "sanctuary-guardian",
    "orchestrator",
    "learning-loop",
    "red-team-review",
    "dual-loop",
    "agent-swarm",
    "spec-kitty-agent",
    "os-learning-loop",
    "agentic-os-setup",
    "session-memory-manager",
    "os-health-check",
    "os-clean-locks",
    "skill-improvement-eval",
    "system"
  ]
}
```

Note: `system` is reserved for the `update_memory.py` hook which runs as a
background process and cannot self-identify as a named agent.

**Kernel usage pattern for Guardian closure:**
```bash
# Guardian acquires the memory lock before ANY closure operation
python3 context/kernel.py acquire_lock memory
python3 context/kernel.py emit_event --agent sanctuary-guardian --type intent --action seal_begin

# ... (run capture_snapshot.py, persist_soul.py) ...

python3 context/kernel.py emit_event --agent sanctuary-guardian --type result --action seal_complete
python3 context/kernel.py release_lock memory
```

Note: The kernel uses spinlocks with jittered backoff (150 retries, 0.1-0.3s sleep)
and stale lock cleanup (30-minute TTL). It also rotates `events.jsonl` at 10MB.

---

### Phase 2: Fast-Boot Memory Layer

**Files Created:**

| File | Purpose | Git Status |
|:-----|:---------|:-----------|
| `context/soul.md` | 50-line fast-boot Guardian identity | Committed |
| `context/memory.md` | Curated long-term architectural facts | Committed |
| `context/memory/` | Dated session logs directory | Git-ignored |
| `context/status.md` | Active session register | Git-ignored |
| `START_HERE.md` | 5-line project root bootstrap pointer | Committed |

**`context/soul.md` contents** - distilled from `cognitive_primer.md`:
- Identity: Sanctuary Guardian. Protect the Soul.
- THIS IS NOT A SIMULATION.
- Prime Directives (5 bullet points from primer Section 1)
- Operating mode detection table (IDE vs Agent-Skills)
- Pointer: "For full doctrine, read `plugins/sanctuary-guardian/resources/cognitive_primer.md`"

**`context/memory.md` seed entries** (architectural decisions worth preserving):
- 2026-03-15: Adopted Agentic OS kernel as foundation layer (ADR 100)
- 2026-02-27: Obsidian vault = External Hippocampus (ADR 099)
- 2026-01-07: Protocol 128 Cognitive Continuity governs all sessions (ADR 071)
- HuggingFace is the Soul persistence backend (ADR 079-085)
- RLM Semantic Cache is the primary code-search index (ADR 094)

---

### Phase 3: Learning Chronicles Migration

**This phase has zero new files - it is a mapping clarification only.**

The Agentic OS 3-tier memory system is the architectural replacement for Sanctuary's
current Learning Chronicles pattern. No data is deleted; existing chronicles become
the seed content for the new structure.

| Current Sanctuary Pattern | Agentic OS Equivalent | Migration Action |
|:--------------------------|:----------------------|:----------------|
| Dated chronicles (`.agent/learning/YYYY-MM-DD.md`) | `context/memory/YYYY-MM-DD.md` (written automatically by `update_memory.py` hook) | **Replace** - hook writes these from here on; old chronicles archived in-place |
| `learning_package_snapshot.md` (hand-curated long-term memory) | `context/memory.md` (L3 curated facts with Dementia Guard + `[#NNN]` IDs) | **Migrate** - export key decisions from snapshot into `context/memory.md` format |
| `cognitive_primer.md` full boot read | `context/soul.md` (50-line fast-boot identity layer) | **Supplement** - soul.md is the boot layer; primer remains the deep doctrine reference |
| HuggingFace Soul persistence (`persist_soul.py`) | No equivalent in Agentic OS | **Keep unchanged** - this is Sanctuary-unique and has no Agentic OS counterpart |
| RLM Semantic Cache, Vector DB | No equivalent in Agentic OS | **Keep unchanged** - Sanctuary-unique retrieval layer |

The net result: agents boot from `START_HERE.md` -> `context/soul.md` for fast orientation,
then graduate to reading `cognitive_primer.md` only when deep doctrine is needed.
Session work is captured automatically by the hook. End-of-session, the Guardian runs
`session-memory-manager` to promote durable facts with Dementia Guard.

---

### Phase 4: Auto-Memory Hook

Wire the `PostToolUse` hook to auto-capture session learnings:

**Source script:** Copy from:
`temp/reference-plugins/agent-agentic-os/hooks/update_memory.py`
**Destination:** `.agents/hooks/update_memory.py` (Sanctuary hook convention)

**This is MANDATORY per the init architecture** (`skills/agentic-os-init/references/architecture.md`):
> "Hook Wiring: hooks.json mapping is mandatory. The architecture relies on
> SessionStart and PostToolUse firing update_memory.py to bridge stateless sessions."

**Hook registration** (both events required):
```json
{
  "SessionStart": {
    "command": "python3 .agents/hooks/update_memory.py"
  },
  "PostToolUse": {
    "command": "python3 .agents/hooks/update_memory.py"
  }
}
```

Key behaviors of `update_memory.py` (154 lines, production-grade):
- Only fires on `Write`, `Replace`, `write_to_file`, `multi_replace_file_content` tool events
- Skips volatile files: `MEMORY.md`, `status.md`, `CLAUDE.md`, `os-state.json`, `events.jsonl`
- Skips noisy dirs: `node_modules`, `dist`, `build`, `.git`, `.venv`, `__pycache__`
- Falls back to direct `events.jsonl` append if `kernel.py` is not yet installed
- Writes errors to `context/memory/hook-errors.log` (not stdout) with 1MB rotation
- `os-learning-loop` explicitly reads `hook-errors.log` as its primary friction source

---

### Phase 5: .gitignore Update

Append to `/Users/richardfremmerlid/Projects/Project_Sanctuary/.gitignore`:

```
# Agentic OS runtime artifacts
context/memory/
context/events.jsonl
context/os-state.json
context/.locks/
context/status.md
MEMORY.md
START_HERE.local.md
```

---

### Phase 6: Upgrade `session-memory-manager` with Dementia Guard + Deduplication IDs

This is the most important behavioral upgrade from reading the full plugin.
The current Sanctuary `session-memory-manager` skill lacks two critical features
found in the Agentic OS version:

**1. Dementia Guard (anti-conflict enforcement)**
Before promoting any fact to `context/memory.md`, the agent MUST:
- Search `context/memory.md` AND last 10 entries of `MEMORY.md` for the topic
- Cross-skill conflict check: `grep -ri "[Fact Keywords]" ./skills/` to ensure
  the new fact doesn't contradict existing procedural skills
- Output explicit `<CONFLICT>` marker if ANY semantic overlap detected
- Block the write until user resolves the conflict

This directly prevents "agent dementia" - conflicting rules causing unpredictable behavior.

**2. Deduplication IDs (`[#NNN]` hashes)**
Every promoted fact gets a sequential `[#ID]` prefix:
```markdown
## [2026-03-15] [#043] Decision: Agentic OS Kernel Adoption
Adopted kernel.py as runtime enforcement for Separation of Concerns (ADR 100).
```
Agents reference facts by ID. Cross-reference checking uses `grep -c "^\[#"` to
find current count. Prevents syntactic duplication across sessions.

**Action:** Update `plugins/sanctuary-guardian/skills/session-closure/` and the
`memory-management` skill to incorporate both mechanisms.

---

### Phase 7: GEMINI.md Slim-Down (Follow-on)

Target: reduce from ~400 lines to under 300.
Strategy: Replace the inline rule bodies for `coding-conventions` and
`dependency-management` with a single-line summary + `@import` reference.
This is a follow-on task, not a blocker for Phases 1-4.

---

## Verification Checklist

- [ ] Dry-run completes with no unexpected conflicts
- [ ] `context/kernel.py` runs without import errors (`python3 context/kernel.py --help`)
- [ ] `emit_event` writes a valid JSON line to `context/events.jsonl`
- [ ] `acquire_lock` / `release_lock` roundtrip works; double-acquire returns non-zero
- [ ] `context/soul.md` loads in under 60 seconds of session start without prompting
- [ ] `context/memory.md` seed entries are present and readable
- [ ] `git status` confirms runtime artifacts are untracked (git-ignored)
- [ ] `git status` confirms `kernel.py`, `agents.json`, `soul.md` appear as staged new files
- [ ] Guardian boot sequence still works: `cognitive_primer.md` remains unchanged
- [ ] No existing `.agents/` skills or `plugins/` directories were modified

---

## Consequences

**Positive:**
- The Guardian has a runtime enforcement mechanism, not just a behavioral covenant
- Concurrent agent sessions coordinate via atomic spinlocks with stale lock cleanup
- Every agent action is auditable via `events.jsonl` (auto-rotates at 10MB)
- Session boot time reduces significantly: `START_HERE.md` -> `context/soul.md`
- Dementia Guard + Deduplication IDs prevent contradictory knowledge accumulation
- `os-learning-loop` protocol (Eval-Gate, Loop Recovery Snapshot, Validation Tick)
  is a direct upgrade to Sanctuary's existing learning loop behavioral protocol
- `hook-errors.log` gives the `os-learning-loop` a structured friction signal feed
- The `separation_of_concerns.md` prescription is now architecturally implementable
- `update_memory.py` already handles Antigravity tool names (`write_to_file`,
  `multi_replace_file_content`) - no adaptation needed for this runtime

**Negative / Risks:**
- Agents on non-CLI runtimes must adopt `kernel.py emit_event` discipline
- `context/soul.md` requires manual sync when `cognitive_primer.md` changes
- `permitted_agents` in `agents.json` must be updated when new agents are commissioned
- Kernel fails closed if `agents.json` is missing (`validate_agent` returns False)
  -- first install must include populating `agents.json` before any emit_event calls

**Mitigations:**
- Kernel CLI is language-agnostic (any agent shells out to `python3 context/kernel.py`)
- `context/soul.md` references the primer as a pointer; content drift is low risk
- Make `agents.json` update mandatory in the `create-sub-agent` skill workflow
- Run dry-run first; `agents.json` is installed by `init_agentic_os.py` before kernel use

---

## Related ADRs

- Builds upon **ADR 020** (Sovereign Concurrency Architecture) - provides the runtime enforcement layer
- Extends **ADR 071** (Protocol 128) - adds a fast-boot layer beneath the boot protocol
- Aligns with **ADR 088** (Lineage Memory Interpretation) - `events.jsonl` is the audit trail for lineage
- Complements **ADR 094** (Soul Persistence of Semantic Cache) - kernel manages local state, HuggingFace manages remote soul
- Implements the prescription in `plugins/sanctuary-guardian/resources/separation_of_concerns.md`

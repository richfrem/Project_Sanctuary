# Loop Architecture: How the Tool-Smith Operates

**Version**: 1.0 | **Last Updated**: 2026-03-14

---

## The Four Layers

```
┌─────────────────────────────────────────────────────┐
│  STRATEGIC LAYER  (Claude - planning only)          │
│  vision.md → roadmap.md → active-sprint.md          │
│  Updates: every 5 keeps or on phase transition      │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  ROUTING LAYER  (Claude - gap classification)       │
│  Reads gap type → selects loop pattern              │
│  Zero Copilot tokens used here                      │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  GENERATION LAYER  (Copilot CLI - gpt5-mini ONLY)  │
│  Bundled context → candidate script → /tmp/         │
│  All code generation happens here. Never Claude.    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  VALIDATION LAYER  (Python scripts - local)         │
│  sandbox_runner → evolution_metrics → keep/discard  │
│  loop_retrospective → friction_log update           │
└─────────────────────────────────────────────────────┘
```

---

## Token Budget Rule

**Claude (me) handles ONLY:**
- Reading files and understanding state (minimal)
- Strategic planning: vision, roadmap, sprint updates
- Gap classification and loop routing decisions
- Writing planning docs (vision.md, roadmap.md, etc.)
- Retrospective synthesis

**Copilot CLI (gpt5-mini) handles EVERYTHING ELSE:**
- All Python script generation (toolbox tools, loop scripts, health checkers)
- All context that can be delegated
- Prompt: always fed via context bundle, never ad-hoc

**Rule**: If it's generatable code, delegate it. If it's strategy or judgment, Claude handles it.

---

## Loop Type Routing Table

| Gap Type | Loop Pattern | Why |
|----------|-------------|-----|
| New toolbox script < 100 lines | `dual-loop` (this loop directly) | Fast, sandboxed, low risk |
| Research / pattern mining | `learning-loop` skill | Knowledge extraction, no code needed |
| Architecture decision | `red-team-review` skill | Adversarial validation before committing |
| Multiple independent scripts | `agent-swarm` skill | Parallel generation |
| Ambiguous / complex gap | `orchestrator` skill | Routes to correct pattern |
| Planning / vision work | Claude directly | Judgment required, no generation |

---

## Context Bundling Strategy

Every Copilot CLI generation call uses a **pre-built context bundle** from `planning/context-bundles/`.
Never pipe a raw file ad-hoc. Always use the manifest.

### Bundle Types

| Bundle | File | Purpose |
|--------|------|---------|
| Tool generation | `context-bundles/tool-generation.json` | Style + sandbox rules + fitness criteria + spec |
| Skill audit | `context-bundles/skill-audit.json` | SKILL.md schema + compliance rules + examples |
| Architecture | `context-bundles/architecture.json` | Plugin structure + ADRs + design patterns |
| Retrospective | `context-bundles/retrospective.json` | results.tsv + friction_log + loop scripts |

### Bundle Generation Workflow

```bash
# 1. Build the bundle from manifest
python3 scripts/bundle_builder.py --manifest planning/context-bundles/tool-generation.json \
    --output /tmp/context_bundle.md

# 2. Pipe bundle + spec to Copilot CLI
cat /tmp/context_bundle.md | copilot -p "
[CONSTRAINT: Output ONLY raw Python. No markdown. No preamble. Target < 80 lines.]
[SPEC: <gap description here>]
" > /tmp/<gap-slug>.py
```

---

## Retrospective Protocol (Every Iteration)

After every KEEP or DISCARD, before picking the next gap:

```
1. What was the friction in this iteration?
   - Over-generation? (add line limit to bundle)
   - Spec too wide? (narrow to 3-4 requirements max)
   - Fitness failure pattern? (log to friction_log.md)
   - Sandbox violation? (update generation constraints)

2. Can I improve a loop infrastructure script right now?
   - If yes: add gap to gaps.json (priority 9+, type=meta)
   - If small fix (< 10 lines): apply inline, log to friction_log.md

3. Does the roadmap need updating?
   - If phase exit criteria just met: update roadmap.md, notify user
   - If new strategic insight: update vision.md
```

---

## Hard Stop Triggers (Always Halt Loop)

1. Script > 100 lines generated → flag to user, narrow spec, retry once
2. `HUMAN_GATE_REQUIRED` from commit_tool.py → halt, report, wait
3. 3 consecutive crashes on same gap → discard gap, add new narrower gap
4. `GAP_NONE` returned twice → run gap_auto_detector, then halt and report
5. User says Stop/Wait → immediate halt, report current state

---

## Meta-Improvement Gaps (Always High Priority)

Gaps prefixed with `meta-` improve the loop itself and always get priority >= 9:
- `meta-bundle-builder` - script to generate context bundles from manifests
- `meta-retrospective` - post-iteration friction logger and proposal generator
- `meta-results-cleaner` - fixes malformed rows in results.tsv
- `meta-gap-scorer` - re-scores gaps based on roadmap phase alignment

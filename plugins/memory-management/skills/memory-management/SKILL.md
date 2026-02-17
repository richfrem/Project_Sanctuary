---
name: memory-management
description: "Tiered memory system for cognitive continuity in Project Sanctuary. Manages hot cache (cognitive_primer.md, guardian_boot_digest.md) and deep storage (LEARNING/, ADRs/, protocols). Use when: (1) starting a session and loading context, (2) deciding what to remember vs forget, (3) promoting/demoting knowledge between tiers, (4) user says 'remember this' or asks about project history, (5) managing the learning_package_snapshot.md hologram."
---

# Memory Management

Tiered memory system that makes the Guardian a continuous collaborator across sessions.

## Architecture

```
HOT CACHE (always loaded at boot)
├── cognitive_primer.md          ← Role, identity, constraints
├── guardian_boot_digest.md      ← Tactical status, active tasks
├── guardian_boot_contract.md    ← Immutable constraints
└── learning_package_snapshot.md ← Cognitive Hologram (1-line per file)

DEEP STORAGE (loaded on demand)
├── LEARNING/topics/             ← Research by topic
│   └── {topic}/analysis.md     ← Deep dives
├── LEARNING/calibration_log.json← Model calibration data
├── ADRs/                        ← Architecture decisions
├── 01_PROTOCOLS/                ← Operational protocols
└── data/soul_traces.jsonl       ← Persistent soul (HuggingFace)
```

## Lookup Flow

```
Query arrives → 
1. Check hot cache (boot files)         → Covers ~90% of context needs
2. Check LEARNING/topics/               → Deep knowledge by subject
3. Check ADRs/                          → Architecture decisions  
4. Query RLM cache (query_cache.py)     → Tool/script discovery
5. Ask user                             → Unknown? Learn it.
```

## Promotion / Demotion Rules

### Promote to Hot Cache when:
- Knowledge is referenced in 3+ consecutive sessions
- It's critical for active work (current spec, active protocol)
- It's a constraint or identity anchor

### Demote to Deep Storage when:
- Spec/feature is completed and merged
- Protocol is superseded by newer version
- Topic research is concluded
- ADR is ratified (move from draft to archive)

### What Goes Where

| Type | Hot Cache | Deep Storage |
|------|-----------|-------------|
| Active tasks | `guardian_boot_digest.md` | — |
| Identity/role | `cognitive_primer.md` | — |
| Constraints | `guardian_boot_contract.md` | — |
| Session state | `learning_package_snapshot.md` | `soul_traces.jsonl` |
| Research topics | Summary in snapshot | `LEARNING/topics/{name}/` |
| Decisions | Referenced by number | `ADRs/{number}_{name}.md` |
| Protocols | Referenced by number | `01_PROTOCOLS/{number}_{name}.md` |
| Tools | — | `rlm_tool_cache.json` |
| Calibration | — | `calibration_log.json` |

## Session Memory Workflow

### At Session Start (Boot)
1. Load hot cache files in order (see learning-loop skill)
2. Iron Check validates snapshot integrity
3. If snapshot stale → flag for refresh at session end

### During Session
- **New learning** → Write to `LEARNING/topics/{topic}/`
- **New decision** → Create ADR draft
- **New tool** → Register in tool inventory
- **Correction** → Update relevant file + note in `disputes.md` if contradicting

### At Session End (Seal)
1. Update `learning_package_snapshot.md` with new content
2. Seal validates no drift since last audit
3. Persist to HuggingFace (soul traces)

## Conventions
- **Hot cache target**: ~200 lines total across boot files
- **Snapshot**: 1 sentence per file, machine-readable
- **Topic folders**: `lowercase-hyphens/`
- **ADR numbering**: 3-digit, sequential
- **Always capture** corrections and contradictions in `disputes.md`

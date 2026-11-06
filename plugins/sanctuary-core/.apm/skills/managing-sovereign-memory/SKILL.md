---
name: managing-sovereign-memory
plugin: sanctuary-core
description: >
  Operates the zero-dependency, filesystem-native 3-layer sovereign memory engine.
  Provides progressive disclosure retrieval (Tier 1 index, Tier 2 summary, Tier 3 deep),
  asymmetric persistence of friction logs across rollbacks, 30-day confidence decay
  auditing, associative wikilink graph traversal, and agent-native cognitive backlog
  and mission management for autonomous AI agents.
allowed-tools: Bash, Read, Write
---

# Managing Sovereign Memory (`managing-sovereign-memory`)

Agent-native memory management system providing token-efficient retrieval, asymmetric recording, and confidence lifecycle governance without background daemons.

## Contents

- [Constraints](#constraints)
- [Quick start](#quick-start)
- [Workflow](#workflow)
- [Verification](#verification)
- [References](#references)

## Constraints

- **Lessons vs procedures**: short evidenced lessons (trigger, action, limits, recorded uses) come from the shared `evaluating-agent-outcomes` (`lessons.py`, store `.agent/learning/events.jsonl`); `LEARNING/procedures/` is only for protected, versioned playbooks with a recorded need. Mission verdicts stay here (`record-result`); engineering delivery is never a verdict (`evaluating-agent-work/references/shared-loop-adapter.md`).

1. **Token Economy**: Never load full Tier 3 files into context when a Tier 1 or Tier 2 summary answers the query.
2. **Asymmetric Invariant**: Code mutations may be rolled back, but architectural friction, edge-case discoveries, and failure logs are NEVER rolled back.
3. **Decay Half-Life**: Knowledge marked `CONFIRMED` not verified within 30 days automatically decays to `OBSERVED`.
4. **Standard Library Only**: Core memory lookups, graph traversal, and recording scripts must never depend on external database daemons or vector services.

## Quick start

```bash
# Retrieve summary of a concept (Tier 2, < 200 tokens)
python3 scripts/disclose.py --level summary --file LEARNING/concepts/sovereign_memory_engine.md

# Traverse associative graph links
python3 scripts/query_graph.py --concept sovereign_memory_engine

# Record an insight or friction log asymmetrically
python3 scripts/record_insight.py --topic "git_guards" --type friction --insight "Pre-commit hook requires UTF-8"

# Audit confidence decay across LEARNING/ concepts
python3 scripts/audit_decay.py --check

# Manage agent-native cognitive backlog (Tier 1 index & mission tracker)
python3 scripts/manage_backlog.py list --format jsonl
python3 scripts/manage_backlog.py compile
```

## Workflow

1. **Orient & Explore (Tier 1)**: Inspect high-level tables via `disclose.py --level index` or `LEARNING/INDEX.md`.
2. **Validate (Tier 2)**: Query executive summaries (`disclose.py --level summary`) to confirm relevance before loading full content.
3. **Deep Retrieval (Tier 3)**: Load full files only when implementing changes or verifying technical contracts.
4. **Asymmetric Recording**: When an experiment or test fails, run `record_insight.py` BEFORE reverting git commits.
5. **Decay Auditing**: Run `audit_decay.py` during Protocol 128 audits to ensure doctrine matches empirical reality.

## Verification

Run the automated test suite:

```bash
python3 -m unittest discover -s tests
```

## References

- [Progressive Disclosure](references/progressive-disclosure.md) — 3-tier retrieval to preserve context tokens.
- [Asymmetric Persistence](references/asymmetric-persistence.md) — Protocol for logging failures before code rollback.
- [Confidence Decay](references/confidence-decay.md) — Epistemic state lifecycle (HYPOTHESIS, OBSERVED, CONFIRMED).
- [Agent-Native Backlog](references/agent-native-backlog.md) — Token-efficient backlog, Popperian falsification manifests, and single-writer authority.

Completion wiring: `record-result` durably marks the final manifest for preservation; after publishing sources to `origin/main`, `merge_prep.py regenerate` runs incremental local extraction, schema and coverage checks. Guardian startup retries pending capture; resume with `soul_coverage.py closure`. Pending/failed/local-complete status is separate from the scientific verdict. Commit the export separately, then sync remotely only with authorization. See the preserving-soul-traces `references/coverage.md` contract.

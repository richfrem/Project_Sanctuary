# Sovereign 3-Layer Memory Engine

**Status:** CONFIRMED
**Source:** Global APM Skill `memory-management`, ADR 100
**Tags:** ["memory", "architecture", "rlm", "cache", "wiki"]

## Architecture

Project Sanctuary implements a zero-dependency, filesystem-native 3-Layer Memory Engine based on Google WikiSkill and Stanford graph-planning principles:

1. **Layer 1: Runtime Context (`context/`)**
   - High-speed bootstrap: `soul.md` (identity), `user.md` (human partner context), `memory.md` (active state).
   - Atomic concurrency and runtime event stream via `context/kernel.py`.
   - Sub-millisecond latency; strict line budgets (<= 100 lines).

2. **Layer 2: Compounding Knowledge Base (`wiki/`)**
   - Atomic concept dossiers tagged with status (`CONFIRMED`, `OBSERVED`, `HYPOTHESIS`).
   - Confidence decay protocol: unverified knowledge decays to `OBSERVED` after 30 days.
   - Asymmetric persistence: failures persist knowledge even when code rolls back.

3. **Layer 3: Immutable Historical Vault (`chronicle/`, `protocols/`, Hugging Face)**
   - Unbroken journal entries in `chronicle/entries/` indexed by `chronicle/ledger.json`.
   - Ratified constitutional doctrines in `protocols/`.
   - Off-box soul traces dataset synchronized to Hugging Face Hub (`Project_Sanctuary_Soul`).

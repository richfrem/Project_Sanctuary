# Progressive Disclosure Protocol

## Contents

- [Overview](#overview)
- [Three-Tier Architecture](#three-tier-architecture)
- [CLI Operations](#cli-operations)
- [Verification](#verification)

---

## Overview

Progressive Disclosure governs how autonomous AI agents query, read, and ingest repository memory while minimizing token consumption and preventing context window bloat.

## Three-Tier Architecture

1. **Tier 1: Index Layer (`--level index`)**
   - High-level bird's-eye map of dossiers, topics, and chapters.
   - Budget: <= 200 tokens.
   - Used for discovery and navigation.

2. **Tier 2: Summary Layer (`--level summary`)**
   - Frontmatter, status tags (`CONFIRMED`, `OBSERVED`), and executive summary.
   - Budget: <= 300 tokens.
   - Used for relevance validation.

3. **Tier 3: Deep Layer (`--level deep`)**
   - Complete technical specification, cryptographic SHA-256 hashes, and code blocks.
   - Budget: Full document.
   - Loaded strictly when targeted edits or deep audits are required.

## CLI Operations

```bash
# Retrieve repository index
python scripts/disclose.py --level index --target LEARNING/

# Retrieve summary of a concept
python scripts/disclose.py --level summary --file LEARNING/concepts/sovereign_memory_engine.md

# Retrieve full document
python scripts/disclose.py --level deep --file protocols/128_Hardened_Learning_Loop_v3.0.md
```

## Verification

Verify extraction speed is sub-10ms using Python standard library:
```bash
python scripts/disclose.py --level summary --file LEARNING/concepts/anti_asch_engine.md
```

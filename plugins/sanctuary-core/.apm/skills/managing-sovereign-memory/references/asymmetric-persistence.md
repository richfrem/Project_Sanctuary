# Asymmetric Persistence Protocol

## Contents

- [Core Mandate](#core-mandate)
- [The Asymmetric Invariant](#the-asymmetric-invariant)
- [Workflow](#workflow)
- [CLI Operations](#cli-operations)

---

## Core Mandate

The fundamental flaw in traditional git-based workflows is that when an experiment fails, `git reset --hard` or deleting a branch destroys both the broken code and the memory of *why* it failed. As a consequence, subsequent agents repeat the identical failure ("The Great Amnesia").

## The Asymmetric Invariant

**Code mutations may be rolled back, but architectural friction, edge-case discoveries, and failure logs are NEVER rolled back.**

```
[Failed Experiment]
         │
         ├──► 1. Asymmetric Record (Insight written to LEARNING/ or memory.md)
         │
         └──► 2. Code Reversion (git checkout / git branch -D)
```

## Workflow

1. Agent encounters unexpected compiler error, broken dependency, or test failure.
2. Before rolling back code, agent runs `record_insight.py` with `--type friction`.
3. The insight is appended to `context/memory.md` and `LEARNING/friction/` or `LEARNING/concepts/`.
4. The failed git branch is discarded; the lesson survives permanently.

## CLI Operations

```bash
# Record friction
python scripts/record_insight.py --topic "git_submodules" --type friction --insight "Sparse checkouts cause path resolution failure in WSL"

# Record confirmed discovery
python scripts/record_insight.py --topic "ripgrep_vs_vector" --type confirmed --insight "Ripgrep achieves sub-5ms exact match with zero daemon overhead"
```

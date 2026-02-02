# Protocol 128 Workflow Alignment Gaps

## Discovered Issues (2026-02-02 Research Walkthrough)

### 1. Phase Count Mismatch

| Source | Phase Count | Notes |
|--------|-------------|-------|
| `workflow-learning-loop.md` | 7 phases (I-VII) | Text workflow |
| `protocol_128_learning_loop.mmd` | 10 phases (I-X) | Diagram |

**Diagram Phases Not in Workflow Text:**
- Phase VIII: Self-Correction (Deployment, Retro, ShareRetro)
- Phase IX: Relational Ingestion & Closure (Ingest, GitOps, End)
- Phase X: Phoenix Forge (Fine-tuning)

**Recommendation:** Either:
1. Add missing phases to `workflow-learning-loop.md`, OR
2. Mark Phases VIII-X as "Optional/Advanced" in the diagram

### 2. Phase Numbering Fixed

**Before Fix (subgraph ID → label mismatch):**
- `PhaseVII` labeled "Phase VIII"
- `PhaseVIII` labeled "Phase IX"
- `PhaseIX` labeled "Phase X"

**After Fix (aligned):**
- `PhaseVIII` labeled "Phase VIII"
- `PhaseIX` labeled "Phase IX"
- `PhaseX` labeled "Phase X"

### 3. Guardian Wakeup Reminder Gap

During research, I skipped `cortex_guardian_wakeup` because it wasn't prominently called out in the workflow. Consider:
- Adding a reminder/checkpoint earlier in `/workflow-learning-loop`
- Making the bash shim call guardian wakeup automatically

---

## Action Items

- [x] Fix phase numbering in .mmd (done 2026-02-02)
- [ ] Align workflow-learning-loop.md with full 10-phase diagram
- [ ] Consider adding guardian wakeup to bash shim

# Phase-1 handoff

No experimental scoring has run. Read preregistration.md, v2/errata.md,
kill-density.md, representation-proof.md and test-output.txt. Pools remain exactly
as sealed before mutant recipes; curation is exhaustive and separate from arms.

Before phase 2, the orchestrator must review and commit freeze.json together with
ALL files it hashes. Do not regenerate pools, edit sources or tune seeds. From
the repository root, verify the committed inputs with:

```bash
python3 -B LEARNING/experiments/mission-041/v3/freeze_tools.py --verify
```

After that commit, the future scoring command is:

```bash
python3 -B LEARNING/experiments/mission-041/v3/run.py --phase2 --freeze-commit FULL_COMMIT_HASH
```

Replace FULL_COMMIT_HASH with the complete committed-freeze hash. This is an
orchestrator attestation, not a local git-history check. The worker has issued no
git command. Phase-2 outputs are append-only named exceptions to inventory.

Cautions: single-author procedural holdout, synthetic correlated faults, some
individual dense faults near a random ceiling, and migration's set-valued domain.
Those limits are registered, not grounds for selecting survivors or retuning.
Installed APM refresh and official budget/contribution ledger accounting remain
orchestrator responsibilities. No measured session token count is available.

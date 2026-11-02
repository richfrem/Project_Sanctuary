# Confidence Decay Protocol

## Contents

- [Epistemic Grounding](#epistemic-grounding)
- [Decay Lifecycle](#decay-lifecycle)
- [Audit & Maintenance](#audit--maintenance)

---

## Epistemic Grounding

To prevent stale consensus and synthetic stagnation, knowledge within the sovereign wiki is treated as an active epistemic state with an expiration date. 

A claim that was true 6 months ago may no longer be true after upstream dependencies or compiler versions change.

## Decay Lifecycle

1. **`HYPOTHESIS`**: Speculative idea or proposal formulated during research.
2. **`OBSERVED`**: Empirically seen or recorded during execution; not yet formally verified through automated testing or compiler gates.
3. **`CONFIRMED`**: Rigorously verified against empirical reality, test suites, and SHA-256 integrity checks.
4. **Decay Half-Life (30 Days)**:
   - Any doctrine marked `CONFIRMED` that is not re-tested or re-verified within 30 days automatically decays to `OBSERVED`.

## Audit & Maintenance

Run the decay auditor periodically or during Protocol 128 audits:

```bash
# Check for decayed entries without mutating
python scripts/audit_decay.py --check

# Apply decay downgrades in-place
python scripts/audit_decay.py --apply
```

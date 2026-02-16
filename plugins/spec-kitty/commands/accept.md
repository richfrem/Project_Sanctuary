---
description: Validate feature readiness — all WPs must be done
argument-hint: "[--feature <SLUG>]"
---

# Accept Feature

Run from **Main Repo Root** after all WPs are done.

```bash
cd <PROJECT_ROOT>
spec-kitty accept --feature <FEATURE-SLUG> --mode local --actor "<AGENT-NAME>"
```

**PROOF**: Output must show `summary.ok: true`.

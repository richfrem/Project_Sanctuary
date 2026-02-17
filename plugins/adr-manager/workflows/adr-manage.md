---
description: Creates a new Architecture Decision Record (ADR) with proper numbering and template.
---

## Phase 0: Pre-Flight (MANDATORY)
```bash
python tools/cli.py workflow start --name codify-adr --target "[Title]"
```
*This aligns with Constitution, determines work type, and initializes tracking.*

---

**Steps:**

1. **Get Sequence Number:**
   Run the following command to find the next available ADR number:
   ```bash
   python plugins/adr-manager/scripts/next_number.py --type adr
   ```
   *Result*: `NNNN` (e.g., `0005`)

2. **File Creation:**
   Create a new file at `ADRs/NNNN-[Title].md`.
   *Example*: `ADRs/0005-use-postgres.md`

3. **Template:**
   Copy contents from: `.agent/templates/outputs/adr-template.md`

   Or manually structured as:
   ```markdown
   # ADR-NNNN: [Title]

   ## Status
   [Proposed | Accepted | Deprecated | Superseded]
   ...
   ```

4. **Confirmation:**
   Inform the user that the ADR has been created and is ready for editing.

---

## Universal Closure (MANDATORY)

### Step A: Self-Retrospective
```bash
/sanctuary-retrospective
```
*Checks: Smoothness, gaps identified, Boy Scout improvements.*

### Step B: Workflow End
```bash
/sanctuary-end "docs: create ADR [Title]" ADRs/
```
*Handles: Human review, git commit/push, PR verification, cleanup.*
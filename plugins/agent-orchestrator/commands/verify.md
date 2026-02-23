# /agent-orchestrator_verify — Verification & Correction Loop

**Purpose:** This command closes the loop on an Inner Loop execution. You inspect the implementation against the original Strategy Packet and the architectural standards of the project.

---

## 🔍 The Verification Protocol

1. **Trigger Automated Audit**:
   ```bash
   /agent-orchestrator_verify --wp WP-NN
   ```
   *Action:* Inspects the worktree for:
   - Files modified vs Scope.
   - Lint/Format compliance.
   - Proof of verification in command output.

2. **The Decision Gate**:

   ### ✅ PASS (Acceptance)
   - **Commit**: `git add . && git commit -m "feat(WP-NN): implementation description"`
   - **Promote Lane**:
     ```bash
     /spec-kitty.review WP-NN --approve
     ```
   - **Proceed**: Move to the next WP or start the Feature Merge.

   ### ❌ FAIL (Correction)
   - **Feedback**: Generate a **Correction Packet** detailing exactly what failed (criteria, bugs, or rule violations).
     ```bash
     /agent-orchestrator_verify --feedback "Description of failure"
     ```
   - **Loop**: Re-delegate the Correction Packet to the Inner Loop.

---

## 📈 Quality Gates (Recursive Audit)
- If the implementation reveals a flaw in the original **Plan**, you MUST return to `/agent-orchestrator_plan` and update the strategy before proceeding.

> [!TIP]
> The correction loop is not a failure—it's the primary mechanism for preserving architectural integrity and "teaching" the Inner Loop patterns.

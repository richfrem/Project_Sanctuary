# Red Team Remediation Plan: Hardening Project Sanctuary

**Verdict**: COMPROMISED
**Objective**: Execute "Path to SECURE" by implementing Improvement Recommendations (IRs).

## 1. Critical Vulnerabilities (Must Fix)

### 🔴 CV-01: Human Gate Bypass (Script Level)
**Risk**: Scripts like `retro.sh`, `seal.sh` could auto-execute state changes.
**Fix (IR-2)**: Add explicit "Press ENTER to confirm" or `read` validation to:
- `scripts/bash/workflow-end.sh` (which calls seal/persist)
- `scripts/bash/workflow-learning-loop.sh`

### 🔴 CV-02: Tool Discovery Fallback Gap
**Risk**: If `query_cache.py` fails, agent might use `grep` "to help".
**Fix (IR-4)**: Update `.agent/rules/01_PROCESS/tool_discovery_enforcement_policy.md` and Constitution to explicitly **PROHIBIT FALLBACK** and mandate `refresh_cache.py`.

### 🔴 CV-03: Ambiguous "State Change"
**Risk**: "Drafting" vs "Writing" ambiguity allows creep.
**Fix (IR-1)**: Add "Definition of State-Changing Operation" to **Constitution Section I** (Supreme Law).

## 2. Structural Weaknesses (High Priority)

### ⚠️ SW-02: Protocol 128 Enforcement
**Risk**: Workflows can end without learning loop.
**Fix (IR-5)**: Update `scripts/bash/workflow-end.sh` to check for a `.sealed` file or similar marker (if feasible) or at least warn user.

### ⚠️ SW-01: Tier Conflict
**Risk**: Technical rules ignore approval gates.
**Fix (IR-12)**: Add "Compliance Checklist" reference to Constitution.

## 3. Execution Plan

1.  **Harden Constitution (v3.7)**:
    -   Insert "State-Changing Operation" Definition (IR-1).
    -   Clarify "No Grep" Fallback (IR-4).
    -   Unify "Explicit Approval" language (IR-9).

2.  **Harden Scripts**:
    -   Mod `scripts/bash/workflow-end.sh`: Add Approval Gate + Protocol 128 Check.
    -   Mod `scripts/bash/workflow-learning-loop.sh`: Add Approval Gate.

3.  **Harden Policies**:
    -   Update `tool_discovery_enforcement_policy.md`.

## 4. Verification
-   Run `workflow-end.sh` -> Verify it halts for approval.
-   Review Constitution v3.7 -> Verify definitions are unambiguous.

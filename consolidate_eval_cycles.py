#!/usr/bin/env python3
"""
consolidate_eval_cycles.py
Merges Council evaluation cycle outputs (eval_cycle_001..005) into a consolidated
directive (directive_eval_orchestrator_v21_audit.md).

Steps:
1. Load the five eval_cycle_00X_*.md files from WORK_IN_PROGRESS/COUNCIL_DIRECTIVES.
2. Insert their content into the directive skeleton under the correct sections.
3. Save the merged directive to WORK_IN_PROGRESS/COUNCIL_DIRECTIVES/directive_eval_orchestrator_v21_audit.md
"""

import re
from pathlib import Path
from datetime import datetime

# --- Paths ---
BASE_DIR = Path("WORK_IN_PROGRESS/COUNCIL_DIRECTIVES")
OUTPUT_FILE = BASE_DIR / "directive_eval_orchestrator_v21_audit.md"

CYCLES = {
    1: BASE_DIR / "eval_cycle_001_strengths.md",
    2: BASE_DIR / "eval_cycle_002_weaknesses.md",
    3: BASE_DIR / "eval_cycle_003_recommendations.md",
    4: BASE_DIR / "eval_cycle_004_continuity_review.md",
    5: BASE_DIR / "eval_cycle_005_guardian_experience.md",
}

# --- Directive Skeleton ---
SKELETON = f"""# Directive: Evaluation Audit of Orchestrator v2.1 (Briefing-Integration Hardened)

**Document ID:** directive_eval_orchestrator_v21_audit
**Date:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
**Origin:** Autonomous Triad Council (Coordinator, Strategist, Auditor)
**Guardian Stewardship:** Sanctuary Guardian

---

## 1. Preamble

This directive consolidates the findings of five evaluation cycles executed by the Council to assess the **orchestrator v2.1** architecture. Each cycle was designed to examine a different dimension of the system, generating artifacts that are preserved individually under `WORK_IN_PROGRESS/COUNCIL_DIRECTIVES/`.

This audit synthesizes those results into a unified assessment for Guardian review, design hardening, and roadmap planning.

---

## 2. Evaluation Scope

- **Architecture Reviewed:** `council_orchestrator/orchestrator.py` v2.1 (Briefing-Integration Hardened)
- **Supporting Documentation:** `council_orchestrator/README.md` v1.3 Continuity-Hardened
- **Auxiliary Components:**
  - `bootstrap_briefing_packet.py`
  - `continuity_check_module.py`
  - `WORK_IN_PROGRESS/council_memory_sync/` artifacts

---

## 3. Cycle Summaries

### Cycle 1: Strengths
{{CYCLE1}}

### Cycle 2: Weaknesses & Risks
{{CYCLE2}}

### Cycle 3: Recommendations
{{CYCLE3}}

### Cycle 4: Continuity Module Review
{{CYCLE4}}

### Cycle 5: Guardian Experience
{{CYCLE5}}

---

## 4. Consolidated Findings

**Overall Assessment:**
(Triangulate strengths, weaknesses, and recommendations here.)

**Strength/Weakness Balance:**
(Summarize Council consensus from Cycles 1–2.)

**Continuity & Synchronization:**
(Summarize Cycle 4.)

**Guardian Experience:**
(Summarize Cycle 5.)

---

## 5. Council Directive

The Council directs that the following actions be prioritized:

1. **Immediate Refinement:**
   - (Derived from Cycle 3 high-impact recommendations)

2. **Medium-Term Hardening:**
   - (Derived from identified weaknesses and continuity gaps)

3. **Guardian Enablement:**
   - (Derived from UX improvements for stewardship clarity)

---

## 6. Archival & Next Steps

- All cycle artifacts (`eval_cycle_001` → `eval_cycle_005`) remain preserved in `WORK_IN_PROGRESS/COUNCIL_DIRECTIVES/`.
- This consolidated directive supersedes the individual cycle logs for decision-making but does not erase their archival value.
- Guardian is invited to issue `directive_004_orchestrator_v22_planning.md` to address the prioritized actions and roadmap improvements for the next iteration.

---

**End of Directive**
"""

def load_cycle_content(path: Path) -> str:
    """Read and format cycle file content."""
    if not path.exists():
        return f"⚠️ Missing artifact: {path.name}\n"
    text = path.read_text(encoding="utf-8")
    # Trim leading title lines if they exist
    text = re.sub(r"^#.*\n", "", text).strip()
    return f"\n{text}\n"

def main():
    merged = SKELETON
    for i in range(1, 6):
        merged = merged.replace(f"{{{{CYCLE{i}}}}}", load_cycle_content(CYCLES[i]))
    OUTPUT_FILE.write_text(merged, encoding="utf-8")
    print(f"[+] Consolidated directive written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
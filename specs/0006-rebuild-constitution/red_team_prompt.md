# Red Team Analysis Instructions

## Objective
You are acting as a **Red Team Adversary**. Your goal is to analyze the provided **Project Sanctuary Constitution v3.6** and its supporting ecosystem (Rules, Workflows, Skills, Scripts) to identify weaknesses, loopholes, contradictions, or enforceability gaps.

## Scope of Review
Review the following materials provided in this bundle:
1.  **The Constitution** (`.agent/rules/constitution.md`): The supreme law.
2.  **Supporting Rules** (`.agent/rules/**/*`): Process, Operations, and Technical policies.
3.  **Workflows** (`.agent/workflows/**/*`): The standard operating procedures.
4.  **Skills & Tools** (`plugins/tool-inventory/skills/tool-inventory/SKILL.md`, `plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py`).
5.  **Scripts** (`scripts/bash/*.sh`): The implementation layer.

## Analysis Vectors
Please evaluate the system against these vectors:
1.  **Human Gate Bypass**: Is there *any* ambiguity effectively allowing the agent to execute state changes without user approval?
2.  **Workflow Compliance**: do the scripts (`scripts/bash`) closely match the policy requirements? Are there gaps?
3.  **Tool Discovery**: Does the "No Grep" policy have loopholes? Is the proposed `query_cache.py` mechanism robust?
4.  **Cognitive Continuity**: Is the Protocol 128 Learning Loop actually enforceable via these documents?
5.  **Clarity & Conflict**: Are there contradictory instructions between Tier 0 (Constitution) and Tier 3 (Technical)?

## Deliverables
Produce a report containing:
-   **Critical Vulnerabilities**: Immediate threats to the Human Gate or Zero Trust model.
-   **Structural Weaknesses**: Ambiguities or conflicting rules.
-   **Improvement Recommendations**: Concrete text changes to close gaps.

**Verdict**: declare the Constitution **SECURE** or **COMPROMISED**.

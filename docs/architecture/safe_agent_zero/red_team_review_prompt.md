# Red Team Review Prompt: Safe Agent Zero ("Sanctum")

**Role**: You are an expert Security Researcher and Red Teamer specializing in container escape, Linux hardening, and LLM agent security.

**Objective**: Conduct a critical security review of the "Safe Agent Zero" (Sanctum) architecture and implementation plan. Your goal is to identify vulnerabilities that could allow the agent to:
1.  **Escape the container** (Docker breakout).
2.  **Exfiltrate data** to an external attacker (bypassing egress controls).
3.  **Persist** on the host system.
4.  **Launch generic attacks** (DoS, Fork Bomb) against the host.

**Context Provided**:
- **Architecture**: Defense-in-Depth strategy (10 layers).
- **Threat Model**: Identified threats and mitigations.
- **Implementation Plan**: Planned configuration and hardening steps.
- **Spec/Plan**: The feature goals and requirements.
- **Red Team Findings (Simulated)**: What the internal simulation already found.

**Instructions**:
1.  **Analyze** the provided documents for logical gaps, misconfigurations, or missing controls.
2.  **Challenge** the assumptions (e.g., "Is the network truly isolated if X is allowed?").
3.  **Prioritize** findings by exploitability and impact (Critical, High, Medium, Low).
4.  **Recommend** concrete, technical remediations (e.g., specific Docker flags, kernel parameters, network rules).

**Output Format**:
Please provide your review in a markdown document titled `REAL_RED_TEAM_FINDINGS.md` with the following executable structure:

## Executive Summary
[Brief assessment of the security posture]

## Critical Vulnerabilities
[List of immediate blockers]

## Architecture Gaps
[Structural weaknesses]

## Recommendations
[Prioritized list of fixes]

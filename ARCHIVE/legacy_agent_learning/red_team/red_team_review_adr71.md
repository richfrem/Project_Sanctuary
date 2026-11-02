# Red Team Review Prompt: ADR 071 v3.2 (Protocol 128 Hardened)

**Context:**
We are formalizing **Protocol 128: Cognitive Continuity**. After a failed audit (v3.0), we have hardened the system against "Manifest Blindspots" and established the **Mandatory Agent Plugin Integration Tool Usage** requirement.

**Hardening Measures in v3.2:**
1.  **Strict Shadow Manifest**: The `cortex_capture_snapshot` tool now performs a raw `git status` check. If critical files (ADRs, Protocols, Scripts) are modified but missing from the manifest, the audit packet generation is **ABORTED**. 
2.  **Mandatory Agent Plugin Integration Tool Usage**: Agents are désormais required to use Agent Plugin Integration tools (filesystem-write, cortex-query) to interact with the codebase. This ensures that the system's "internal sensors" remain calibrated.
3.  **Two-Factor Integrity**: We now implement Hard HMAC checks for code/protocols and Soft Semantic checks for learning notes to resolve the "Brittle Boot" scenario.

**Your Mission (Red Team):**
Please audit these new mitigations:

1.  **Shadow Manifest Robustness**: Does the "Shadow Manifest" check in `operations.py` effectively prevent a malicious agent from sneaking changes into the `ADRs/` or `mcp_servers/` directories?
2.  **Agent Plugin Integration Tool Dependency**: Does mandating Agent Plugin Integration tools create a "Circular Trust" vulnerability? What happens if the Agent Plugin Integration tools themselves are compromised?
3.  **Safe Mode Authority**: Is the "Safe Mode" state (Read-only, disabled write) a sufficient containment strategy if an integrity failure is detected?

**Reference Material:**
- `ADRs/071_protocol_128_cognitive_continuity.md` (v3.2)
- `mcp_servers/rag_cortex/operations.py` (Shadow Manifest Logic)
- `LEARNING/topics/mcp_tool_usage.md` (Tooling Protocol)

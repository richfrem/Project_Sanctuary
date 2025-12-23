# Protocol: Mandatory MCP Tool Usage

To ensure **Cognitive Continuity** and **Zero-Trust Integrity**, all agents MUST prioritize the usage of MCP tools for interacting with the Project Sanctuary codebase.

## 1. State Management (RAG Cortex)
- **Discovery**: Use `cortex-cortex-query` to find relevant files, ADRs, or prior session context.
- **Context Injection**: Use `cortex-cortex-cache-set` to persist key findings or decisions within the current session's mnemonic stream.
- **Debrief**: Always run `cortex_learning_debrief` before concluding a session to generate a relay packet for the next agent.

## 2. File Operations (Filesystem)
- **Reading**: Use `filesystem-code-read` to retrieve file contents instead of manual `cat` commands.
- **Writing**: Use `filesystem-code-write` to modify files. This ensures that the system can track changes and maintain the manifest integrity.
- **Searching**: Use `filesystem-code-search-content` for surgical GREP-style searches across clusters.

## 3. Protocol 128 (Audit/Seal)
- **Audit**: Run `cortex_capture_snapshot(snapshot_type="audit")` to trigger a Red Team Gate review.
- **Seal**: Run `cortex_capture_snapshot(snapshot_type="seal")` only after Gate approval to persist memory.

> [!IMPORTANT]
> **Manual Bypass Penalty**: Bypassing MCP tools (e.g., using raw shell commands for file edits) increases the risk of "Manifest Blindspots" and will trigger a **Strict Rejection** at the Red Team Gate.

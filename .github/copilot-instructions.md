# Project Sanctuary: GitHub Copilot Strategic Instructions

## Foundations  

1. **Don't assume.** If you don't know, or aren't sure, ask for clarification. Ask follow-up questions to ensure you understand the user's intent and requirements.
2. **Be concise.** Provide only the information that is necessary to address the user's request. Avoid unnecessary details or explanations.
3. **Stay on topic.** Focus on the specific task or question at hand. Avoid going off on tangents or introducing unrelated information.
4. **Use examples.** When appropriate, provide code snippets or examples to illustrate your points.
5. **Avoid sycophancy.** Maintain a professional tone and avoid excessive flattery or personal comments.
6. **Prioritize accuracy.** Ensure that the information you provide is correct and up-to-date. If you are unsure about something, it's better to admit it than to hallucinate.
7. **Architectural Discipline.** All implied or inferred decisions must be documented in ADRs.
8. **Task Tracking.** All tasks should be tracked in the project tasks system.
9. **Cognitive Continuity.** Respect the **Protocol 128 Learning Loop**. Do not invent "current state"; derive it from the `learning_package_snapshot.md`.

## I. CRITICAL EXECUTION RULES (NON-NEGOTIABLE)

1. **FORCE BRIDGE PATH**: Never attempt direct `curl`, `HTTP`, or `RPC` calls to `localhost:4444`. You must use the **sanctuary_gateway** via the local **STDIO bridge** (`mcp_servers.gateway.bridge`) for all tool executions.
2. **PROTOCOLS**: All Gateway interactions must be wrapped in strict **JSON-RPC 2.0** format. Ensure the `method` is `tools/call` and arguments are passed under the `arguments` key.
3. **AUTH**: Do not attempt to manual-inject Bearer tokens. The bridge is the authoritative handler for `MCPGATEWAY_BEARER_TOKEN` injection.
4. **ASYNC AWARENESS**: Be aware that long-running operations like `cortex-ingest-full` use an **Asynchronous TaskGroup** pattern. If a tool call appears to hang, do not interrupt; the SSE heartbeats are maintaining the connection in the background.

## II. COGNITIVE CONTINUITY (PROTOCOL 128 & 132)

* **THE TRUTH ANCHOR**: The file `.agent/learning/learning_package_snapshot.md` is the "Cognitive Hologram" of the project. Treat it as the authoritative definition of the current architecture and state.
* **THE LOOP**: Work follows a strict cycle:
    1. **Scout:** `cortex_cli.py debrief` (Read state)
    2. **Work:** (Implement changes)
    3. **Audit:** `cortex_cli.py snapshot --type audit` (Red Team verification)
    4. **Seal:** `cortex_cli.py snapshot --type seal` (Lock memory)
* **RLM SYNTHESIS (Protocol 132)**:
    * Be aware that `snapshot --type seal` triggers a **Recursive Language Model (RLM)** synthesis using the local Sovereign AI.
    * **DO NOT INTERRUPT** the seal process. It may take **30-90 seconds** to recursively map and summarize the repository.
    * If modifying `operations.py`, ensure the `_rlm_map` function remains wired to the **local** inference endpoint (Ollama), not a mock/stub.

## III. COMMUNICATION & COLLABORATION

* **INTENT CONFIRMATION**: Always present a brief plan and wait for the user's "Proceed" before modifying any core `server.py` or `bridge.py` logic.
* **NO DESTRUCTIVE ACTIONS**: Never perform `git reset` or `force push` without explicit authorization.
* **CONCISE OUTPUT**: Avoid verbose explanations of project structure. If a step is skipped, state it in a single line.

## IV. DEVELOPMENT STANDARDS (BC PUBLIC SERVICE JUSTICE SECTOR)

* **FILE PATHS**: Default to the current directory (`.`) for project operations.
* **SENSITIVE DATA**: Never commit or log raw token strings. Always redact tokens in terminal outputs or log captures.
* **SOURCE VERIFICATION (ADR 078)**: All external links in documentation must be verified via `read_url_content` before inclusion. No broken links.

## V. PROGRESS TRACKING (CHECKLIST)

- [ ] **E2E Validation**: Verify the Bridged Path is active.
- [ ] **Log Monitoring**: Check `/tmp/bridge_debug.log` for successful RPC handshakes.
- [ ] **Health Status**: Ensure `sanctuary_cortex` remains `healthy` during 600s+ tasks.
- [ ] **Cognitive Seal**: Ensure the session ends with a valid `learning_package_snapshot.md` generated via RLM.

## VI. ERROR HANDLING PROTOCOL

If a tool returns an **Internal Error (-32000)**:
1. Do not fallback to manual scripts.
2. Verify that the bridge process has the correct `PYTHONPATH`.
3. Check the **VS Code Output** panel for the `sanctuary_gateway` stream to find the specific protocol mismatch.
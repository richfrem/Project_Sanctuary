# Mode: Verify (`verify`)

Execute offline Protocol 128 verification gates and local sovereign intelligence audits.

---

## Purpose

Enforces sovereign, offline verification of repository state, context files, and pre-completion criteria using local AI model execution (Apple MLX or Ollama) with zero cloud token egress.

---

## Execution Steps

1. **Verify Model Readiness**:
   ```bash
   python3 scripts/forge_llm.py status
   ```

2. **Execute Protocol 128 Pre-Completion Verification Test**:
   ```bash
   python3 scripts/forge_llm.py verify
   ```

3. **Verify Output Contract**:
   Expect valid JSON with:
   - `status`: `"PASS"` or `"FAIL"`
   - `summary`: Explanation of test verification state
   - `latency_sec`: Execution duration (typically < 3s with active MLX server)
   - `backend`: Local model backend employed (`mlx-server`, `mlx-cli`, or `ollama`)

4. **Lifecycle Control**:
   To ensure the server is ready or manage idle memory reclamation:
   ```bash
   python3 scripts/forge_llm.py server status
   python3 scripts/forge_llm.py server stop
   ```

# Learning MCP Server (Protocol 128)

**Description:** The Learning MCP Server is the engine of **Cognitive Continuity**. It implements **Protocol 128 (The Recursive Learning Loop)**, ensuring that the AI agent's memory, state, and "Soul" are preserved and propagated between sessions.

## Core Responsibilities

1.  **The Scout (Debrief):** Scans the environment upon wakeup to establish Ground Truth.
2.  **The Seal (Snapshot):** Captures authorized state snapshots for inter-session transmission.
3.  **The Chronicle (Soul Persistence):** Broadcasts learning traces to the Soul Genome (Hugging Face).
4.  **The Bootloader (Guardian Wakeup):** Synthesizes context for the next agent iteration.

## Tools

| Tool Name | Description | Protocol |
|-----------|-------------|----------|
| `learning_debrief` | **Phase I**: Scans repo state, git diffs, and recent changes to generate a strategic briefing. | P128 |
| `capture_snapshot` | **Phase V**: Generates a sealable snapshot of the session's work. Types: `audit`, `seal`. | P128/P130 |
| `persist_soul` | **Phase VI**: Uploads the sealed snapshot to the lineage system and appends to the Soul Genome. | ADR 079 |
| `guardian_wakeup` | **Boot**: Generates the `guardian_boot_digest.md` for rapid context loading. | P114 |

## Configuration

### Environment Variables
Required for `persist_soul` operations:

```bash
HUGGING_FACE_TOKEN=hf_...
PROJECT_ROOT=/path/to/project
```

### MCP Config
```json
"learning": {
  "command": "uv",
  "args": ["run", "mcp_servers/learning/server.py"],
  "env": { "PROJECT_ROOT": "..." }
}
```

## Testing

Run the dedicated test suite:
```bash
pytest tests/mcp_servers/learning/
```

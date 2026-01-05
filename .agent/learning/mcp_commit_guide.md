# MCP Commit Message Guide

## Format

```
mcp(<domain>): <description>
```

## Valid Domains

| Domain | Example |
|--------|---------|
| `chronicle` | `mcp(chronicle): create entry #283 - architecture complete` |
| `protocol` | `mcp(protocol): update P115 to v2.0` |
| `adr` | `mcp(adr): create ADR #037 - state machine pattern` |
| `task` | `mcp(task): move #030 to active status` |
| `cortex` | `mcp(cortex): ingest architecture documents` |
| `council` | `mcp(council): create deliberation for strategy` |
| `config` | `mcp(config): update API key for OpenAI` |
| `code` | `mcp(code): implement safety validator module` |
| `git_workflow` | `mcp(git_workflow): create feature/mcp-implementation` |
| `forge` | `mcp(forge): initiate guardian-02 training job` |

## Examples

**Good:**

```
mcp(chronicle): create entry #283 documenting MCP architecture completion
mcp(forge): initiate model training for guardian-02-v1
mcp(git_workflow): create feature branch for task-030
```

**Bad:**

```
mcp: update files  # Missing domain
mcp(invalid): test  # Invalid domain
mcp(chronicle): fix  # Description too short
```

## Migration Period

**Status:** The failed `commit_manifest.json` system and its associated pre-commit hook have been **permanently purged**.

During the transition, both MCP and Legacy commit formats are supported, but all commits now adhere to **Protocol 101 v3.0 (Functional Coherence)**.

  - **MCP commits**:
      - Use the canonical `mcp(<domain>):` format.
      - **No environment variable is required** to bypass the obsolete manifest check.
  - **Legacy commits**:
      - Use conventional commit format (e.g., `FEAT: Add new server`).
      - **The manifest file is not required.**

-----

**Final Action:** Once you have updated this file and the other documentation files, execute the final commit and push to complete the purge:

```bash
git add .
git commit -m "CHORE: Final Purge of Protocol 101 documentation and failed manifest system."
git push origin main
```
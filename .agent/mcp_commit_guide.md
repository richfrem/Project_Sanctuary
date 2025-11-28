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

During migration, both MCP and legacy commit formats are supported:
## Migration Period

During migration, both MCP and legacy commit formats are supported:
- **MCP commits**: 
    - Use `mcp(<domain>):` format.
    - **MUST** set `IS_MCP_AGENT=1` in the environment to bypass the manifest check.
    - Example: `IS_MCP_AGENT=1 git commit -m "mcp(task): update status"`
- **Legacy commits**: Must include `commit_manifest.json` as per Protocol 101.

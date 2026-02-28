# [Server Name] Agent Plugin Integration Server

**Description:** [Brief description of what this server does and its role in the ecosystem]

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `tool_name` | [Description of what the tool does] | `arg1` (type): desc<br>`arg2` (type): desc |

## Resources

| Resource URI | Description | Mime Type |
|--------------|-------------|-----------|
| `resource://uri` | [Description] | `application/json` |

## Prompts

| Prompt Name | Description | Arguments |
|-------------|-------------|-----------|
| `prompt_name` | [Description] | `arg1` |

## Configuration

### Environment Variables
Create a `.env` file in the server root or project root:

```bash
VAR_NAME=value
```

### Agent Plugin Integration Config
Add this to your `mcp_config.json`:

```json
"server_name": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/server_dir",
    "run",
    "server.py"
  ],
  "env": {
    "VAR_NAME": "value"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest mcp_servers/server_dir/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `tool_name` appears in the tool list.
3.  **Call Tool:** Execute `tool_name` with valid arguments and verify output.

## Dependencies

- `mcp`
- [Other dependencies]

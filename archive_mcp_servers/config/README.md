# Config MCP Server

**Description:** The Config MCP server provides tools for managing configuration files within the `.agent/config/` directory. It ensures safe access to configuration data with path validation, automatic backups, and support for JSON/YAML formats.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `config_list` | List all configuration files. | None |
| `config_read` | Read a configuration file. | `filename` (str): Config file name. |
| `config_write` | Write a configuration file with automatic backup. | `filename` (str): Config file name.<br>`content` (str): Content to write. |
| `config_delete` | Delete a configuration file. | `filename` (str): Config file name. |

## Resources

*No resources currently exposed.*

## Prompts

*No prompts currently exposed.*

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Required
PROJECT_ROOT=/path/to/Project_Sanctuary
```

### MCP Config
Add this to your `mcp_config.json`:

```json
"config": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/config",
    "run",
    "server.py"
  ],
  "env": {
    "PYTHONPATH": "${PYTHONPATH}:${PWD}",
    "PROJECT_ROOT": "${PWD}"
  }
}
```

## Testing

### Unit Tests
Run the test suite for this server:

```bash
pytest mcp_servers/config/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `config_list` appears in the tool list.
3.  **Call Tool:** Execute `config_list` and verify it returns config files.

## Architecture

### Overview
This server manages the `.agent/config/` directory.

**Safety Features:**
- ✅ **Safe Path Validation:** Restricts access to `.agent/config/` directory
- ✅ **Automatic Backups:** Creates timestamped backups before overwriting files
- ✅ **Format Support:** Native JSON and YAML handling
- ✅ **Security:** Prevents directory traversal attacks

## Dependencies

- `mcp`

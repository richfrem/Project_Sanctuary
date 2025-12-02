# Code MCP Server

**Description:** The Code MCP server provides tools for code quality operations including linting, formatting, and static analysis. It integrates with popular Python code quality tools while enforcing safety checks.

## Tools

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `code_lint` | Run linting on a file or directory. | `path` (str): Path to check.<br>`tool` (str): Tool (ruff, pylint, flake8). |
| `code_format` | Format code in a file or directory. | `path` (str): Path to format.<br>`tool` (str): Tool (ruff, black).<br>`check_only` (bool): Verify only. |
| `code_analyze` | Perform static analysis on code. | `path` (str): Path to analyze. |
| `code_check_tools` | Check which code quality tools are available. | None |
| `code_find_file` | Find files by name or glob pattern. | `name_pattern` (str): Pattern.<br>`directory` (str): Search root. |
| `code_list_files` | List files in a directory with optional pattern. | `directory` (str): Root.<br>`pattern` (str): Filter.<br>`recursive` (bool): Recursive. |
| `code_search_content` | Search for text/patterns in code files. | `query` (str): Search term.<br>`file_pattern` (str): File filter.<br>`case_sensitive` (bool): Case sensitivity. |
| `code_read` | Read file contents. | `path` (str): File path.<br>`max_size_mb` (int): Size limit. |
| `code_write` | Write/update file with automatic backup. | `path` (str): File path.<br>`content` (str): Content.<br>`backup` (bool): Create backup.<br>`create_dirs` (bool): Create parents. |
| `code_get_info` | Get file metadata. | `path` (str): File path. |

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
"code": {
  "command": "uv",
  "args": [
    "--directory",
    "mcp_servers/code",
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
pytest mcp_servers/code/tests
```

### Manual Verification
1.  **Build/Run:** Ensure the server starts without errors.
2.  **List Tools:** Verify `code_lint` appears in the tool list.
3.  **Call Tool:** Execute `code_check_tools` and verify it returns available tools.

## Architecture

### Overview
This server provides safe file system access and code quality tool integration.

**Safety Features:**
- ✅ **Safe Path Validation:** Restricts access to project directory
- ✅ **Multiple Tool Support:** Ruff, Black, Pylint, Flake8, Mypy
- ✅ **Check-Only Mode:** Verify formatting without modifying files
- ✅ **Security:** Prevents directory traversal attacks

## Dependencies

- `mcp`
- `ruff`
- `black`
- `pylint`
- `flake8`
- `mypy`

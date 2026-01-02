# Cluster: sanctuary_filesystem

**Role:** Secure file system operations and directory management.  
**Port:** 8101  
**Front-end Cluster:** ✅ Yes

## Overview
The `sanctuary_filesystem` cluster isolates all I/O operations from the main agent environment. It is the modern successor to the script-based Code MCP, providing a containerized interface for file manipulation.

## Verification Specs (Tier 3: Bridge)
*   **Target:** Gateway Bridge & RPC Routing
*   **Method:** `pytest tests/mcp_servers/gateway/clusters/filesystem/test_gateway.py`
*   **Backend Source**: Migrated from the `code` MCP server logic.

## Tool Inventory & Legacy Mapping
| Function | Gateway Tool Name | Legacy Operation | T1/T2 Method |
| :--- | :--- | :--- | :--- |
| **Lint** | `sanctuary_filesystem-code-lint` | `code_lint` | `pytest tests/mcp_servers/code/` |
| **Format** | `sanctuary_filesystem-code-format` | `code_format` | |
| **Analyze** | `sanctuary_filesystem-code-analyze` | `code_analyze` | |
| **Check** | `sanctuary_filesystem-code-check-tools`| `code_check_tools`| |
| **Find** | `sanctuary_filesystem-code-find-file` | `code_find_file` | |
| **List** | `sanctuary_filesystem-code-list-files` | `code_list_files` | |
| **Search** | `sanctuary_filesystem-code-search-content`| `code_search_content`| |
| **Read** | `sanctuary_filesystem-code-read` | `code_read` | |
| **Write** | `sanctuary_filesystem-code-write` | `code_write` | |
| **Metadata** | `sanctuary_filesystem-code-get-info` | `code_get_info` | |

## Infrastructure Notes
- **Sandboxing**: Execution happens within the `sanctuary_filesystem` container.
- **Safety**: Integrates with the `sanctuary_allowlist.py` plugin for write protection.

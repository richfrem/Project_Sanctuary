# Task #038: Integrate Task MCP with Claude Desktop

**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 30 minutes  
**Dependencies:** Task #031 (Implement Task MCP)

---

## Objective

Register the Task MCP server with Claude Desktop to enable natural language task management via MCP protocol.

---

## Deliverables

1. Claude Desktop configuration file updated with Task MCP server
2. Task MCP server visible in Claude Desktop
3. Verification that tasks can be created via natural language
4. Documentation in Task MCP README

---

## Acceptance Criteria

- Task MCP server appears in Claude Desktop's MCP servers list
- Can create tasks using natural language (e.g., "Create a task to implement feature X")
- Can list, search, and update tasks via Claude
- All 6 MCP tools are accessible from Claude Desktop

---

## Implementation Steps

### 1. Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "task-mcp": {
      "command": "python",
      "args": ["-m", "mcp_servers.task.server"],
      "env": {
        "PYTHONPATH": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
        "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
      },
      "cwd": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
    }
  }
}
```

### 2. Restart Claude Desktop

- Quit Claude Desktop completely
- Reopen Claude Desktop
- Verify Task MCP server appears in MCP panel

### 3. Test Integration

Try these commands in Claude:
- "Create a task to implement user authentication"
- "List all tasks in backlog"
- "Move task #037 to in-progress"
- "Search for tasks about MCP"

### 4. Verify All Tools Work

Test each of the 6 MCP tools:
- ✅ create_task
- ✅ update_task
- ✅ update_task_status
- ✅ get_task
- ✅ list_tasks
- ✅ search_tasks

---

## How It Works

### Schema Discovery
When Claude Desktop connects to the Task MCP server, it performs a "handshake" where it asks for the list of available tools. The server responds with the JSON schema for each tool (defined in `server.py`).

**Example Schema Sent to Claude:**
```json
{
  "name": "create_task",
  "description": "Create a new task file in TASKS/ directory",
  "inputSchema": {
    "type": "object",
    "properties": {
      "title": { "type": "string" },
      "priority": { "type": "string", "enum": ["Critical", "High", "Medium", "Low"] },
      ...
    },
    "required": ["title", "objective", "deliverables", "acceptance_criteria"]
  }
}
```

### Data Flow
1. **User Intent:** You say "Create a high priority task for X"
2. **Claude Processing:** Claude maps your intent to the `create_task` tool schema.
3. **Tool Call:** Claude sends a JSON request to the MCP server:
   ```json
   {
     "tool": "create_task",
     "arguments": {
       "title": "Implement X",
       "priority": "High",
       ...
     }
   }
   ```
4. **Execution:** The MCP server executes the Python code in `operations.py`.
5. **Response:** The server returns the result (e.g., file path created).

### What Data Claude Needs
Claude only needs your **intent**. It will infer the required fields based on your prompt and the schema.
- **Required:** Title, Objective, Deliverables, Acceptance Criteria (Claude will often generate drafts for these if you provide a strong title/objective).
- **Optional:** Priority, Status, Lead, etc.

---

## Notes

- Task MCP server uses stdio transport (standard for Claude Desktop)
- Server starts automatically when Claude Desktop launches
- Logs available in `~/Library/Logs/Claude/`
- Configuration persists across Claude Desktop restarts

---

**Domain:** `project_sanctuary.document.task`  
**Related:** Task #031 (Implement Task MCP)

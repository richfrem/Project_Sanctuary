# MCP Git Tool Parameter Issue

**Date:** 2025-11-29  
**Discovered During:** Task 055 - Git Operations Verification  
**Status:** 🔍 Needs Investigation

---

## Issue Description

The `git_push_feature` MCP tool has `force` and `no_verify` parameters defined in the server code, but when calling the tool via the MCP interface, it throws a validation error:

```
Unexpected keyword argument [type=unexpected_keyword_argument, input_value=True, input_type=bool]
```

---

## Evidence

### Server Code (Correct)

**File:** `mcp_servers/system/git_workflow/server.py:80`

```python
@mcp.tool()
def git_push_feature(force: bool = False, no_verify: bool = False) -> str:
    """
    Push the current feature branch to origin.
    
    Args:
        force: Force push (git push --force). Use with caution.
        no_verify: Bypass pre-push hooks (git push --no-verify). Useful if git-lfs is missing.
    
    Returns:
        Push status.
    """
    try:
        current = git_ops.get_current_branch()
        if current == "main":
            return "Error: Cannot push main directly via this tool."
            
        output = git_ops.push("origin", current, force=force, no_verify=no_verify)
        return f"Pushed {current} to origin: {output}"
    except Exception as e:
        return f"Failed to push feature: {str(e)}"
```

### MCP Call (Fails)

```python
mcp4_git_push_feature(no_verify=True)
# Error: Unexpected keyword argument
```

### Workaround Used

```bash
git push origin feature/task-055-git-testing-verification
# Works fine (git-lfs hook didn't block in this case)
```

---

## Possible Causes

1. **MCP Server Not Restarted** - The MCP server might need to be restarted after code changes
2. **Configuration Cache** - Claude/Antigravity might be caching an old version of the tool schema
3. **FastMCP Schema Generation** - FastMCP might not be properly detecting the new parameters
4. **MCP Configuration File** - The MCP configuration might need to be updated

---

## Investigation Steps

1. Check if MCP server needs restart
2. Verify FastMCP version and schema generation
3. Check Claude Desktop / Antigravity MCP configuration files
4. Test with other MCP tools to see if parameter changes are recognized
5. Review FastMCP documentation for parameter registration

---

## Impact

**Low** - Workaround available (use direct git commands or `git_push_feature()` without parameters)

The underlying `GitOperations.push()` method works correctly with both `force` and `no_verify` parameters (verified in tests). The issue is only with the MCP layer parameter recognition.

---

## Next Steps

1. Create Task 057 to investigate and fix MCP parameter registration
2. Test if restarting MCP servers resolves the issue
3. Document proper procedure for updating MCP tool signatures

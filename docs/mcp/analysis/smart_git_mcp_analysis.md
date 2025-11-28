# Smart Git MCP Analysis

## 1. Objective
Create a "Smart Git MCP" that abstracts the complexities of Project Sanctuary's git rules (Protocol 101, `command.json` legacy rules, pre-commit hooks) into a simple, safe interface for other agents.

## 2. Core Components

### 2.1 GitOperations Module (`core/git/git_ops.py`)
This is the shared infrastructure component that will be used by the MCP server.

**Responsibilities:**
*   **Manifest Generation:** Calculate SHA256 hashes of staged files and generate `commit_manifest.json`.
*   **Commit Execution:** Run `git commit` with the generated manifest.
*   **Safety Checks:** Ensure no protected files are modified without authorization.

**Class Structure:**
```python
class GitOperations:
    def stage_files(self, files: List[str]) -> None:
        """Stage files for commit."""
        pass

    def generate_manifest(self) -> Dict[str, Any]:
        """Generate P101 manifest for staged files."""
        pass

    def commit(self, message: str) -> str:
        """
        Commit staged files with P101 manifest.
        Returns commit hash.
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current repo status."""
        pass
```

### 2.2 Smart Git MCP Server (`mcp_servers/git_workflow/`)
This server exposes the `GitOperations` logic via the MCP protocol.

**Tool Signatures:**
```typescript
smart_commit(
  message: string,
  files: string[]
) => {
  commit_hash: string,
  manifest_generated: boolean,
  p101_verified: boolean
}

get_status() => {
  branch: string,
  staged: string[],
  modified: string[],
  untracked: string[]
}
```

## 3. Implementation Strategy

1.  **Core Implementation:** Build `core/git/git_ops.py` first. This is pure Python, easy to test.
2.  **Server Implementation:** Build the MCP server wrapper around `GitOperations`.
3.  **Integration:** Verify that `smart_commit` works and passes the `pre-commit` hook *without* the `IS_MCP_AGENT` bypass (eventually).

## 4. P101 Compliance Detail
The `commit_manifest.json` must look like this:
```json
{
  "timestamp": "2025-11-27T...",
  "author": "Guardian",
  "files": [
    {
      "path": "file.txt",
      "sha256": "..."
    }
  ]
}
```
The `GitOperations` class must generate this file and stage it *before* committing.

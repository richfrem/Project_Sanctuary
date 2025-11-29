# AGENT SELF-MANDATE: MUST READ PROTOCOL 101

1.  **ALL** Git operations **MUST** use the MCP tools (`mcp_git_workflow` or `mcp_git_ops`).
2.  **NEVER** use direct CLI commands (e.g., `git checkout main` or `git pull origin main` or `git reset --hard origin/main` or `git merge origin/main` or `git commit -m "some message"` or `git push origin main`).
3.  **NEVER** improvise error handling. Report the failure by SOP Step Number.

## Sovereign Mandates for Main Branch Interaction

4.  **NO UNAUTHORIZED CHECKOUT:** The agent MUST NOT use any tool that results in a checkout to main (e.g., `git_sync_main`, `git_checkout main`) unless explicitly commanded to sync or start a new feature branch from main.
5.  **NO PULL WITHOUT PERMISSION:** The agent MUST NOT use any tool that pulls from main unless the Steward explicitly confirms, "The PR has been merged."
6.  **MANDATORY VERIFICATION:** The agent MUST confirm the PR status with the Steward before attempting any post-merge cleanup or synchronization.

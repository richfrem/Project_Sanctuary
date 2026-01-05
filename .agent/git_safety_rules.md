# Antigravity Git Safety Rules - Project Sanctuary

**Version:** 3.0  
**Last Updated:** 2025-11-29  
**Purpose:** Prevent data loss, ensure functional coherence, and enforce architectural integrity in AI-assisted development.  
**Project Context:** Project Sanctuary with Protocol 101 (Absolute Stability) enforcement.

-----

## 0\. Protocol 101: Absolute Stability (Functional Coherence) ✅

**CRITICAL:** This project enforces **Protocol 101 v3.0: The Doctrine of Absolute Stability**. Commit integrity is verified by **Functional Coherence** (passing the comprehensive automated test suite) and **Action Integrity** (adherence to the Git Whitelist and Prohibition of Destructive Commands). The old `commit_manifest.json` system is **permanently purged and forbidden**.

-----

### 3\. Commit Message Format

**Legacy Format (Human):**

  - Follow Conventional Commits
  - Must clearly describe the change

**MCP Format (Agent):**

  - Format: `mcp(<domain>): <description>`
  - Domains: `chronicle`, `protocol`, `adr`, `task`, `cortex`, `council`, `config`, `code`, `git_workflow`, `forge`

### Commit Workflow Options

#### Option A: Use Council Orchestrator (Recommended)

For automated, Protocol 101-compliant commits (Functional Coherence verified by tests):

```bash
# Create command.json for orchestrator
cat > command.json << 'EOF'
{
  "command_type": "git_operations",
  "git_operations": {
    "files_to_add": [
      "tasks/backlog/025_implement_mcp_rag_tool_server.md",
      ".env.example"
    ],
    "commit_message": "Add Task #025: Implement MCP RAG Tool Server",
    "push_after_commit": false
  }
}
EOF

# Execute via orchestrator 
# The orchestrator is architecturally mandated to run the test suite and verify success
python3 council_orchestrator/app/main.py command.json
```

The orchestrator is now architecturally mandated to:

  - Run `./scripts/run_genome_tests.sh` to confirm **Functional Coherence (Protocol 101)**.
  - Stage files.
  - Commit and optionally push.

#### Option B: Manual Commit (Functional Coherence Required)

For simple commits, **you must manually verify Functional Coherence before committing**:

```bash
# 1. VERIFY FUNCTIONAL COHERENCE (MANDATORY)
./scripts/run_genome_tests.sh

# 2. Stage files
git add tasks/backlog/025_implement_mcp_rag_tool_server.md .env.example

# 3. Commit
git commit -m "Add Task #025: Implement MCP RAG Tool Server"
```

#### Option C: Sovereign Override (Emergencies Only)

**Use ONLY for urgent hotfixes with Guardian approval to bypass local checks:**

```bash
git commit --no-verify -m "Emergency: [description]"
```

-----

This file defines strict rules for Git operations to prevent data loss, unnecessary branching, and workflow confusion. It is designed for use with AI coding assistants like **Antigravity**, **Cursor**, **GitHub Copilot**, and **VS Code extensions**.

## 1\. Branch Management

  * **NO Panic Branching:** Never create a new branch just because a push to `main` or another branch failed.
  * **Single Source of Truth:** Always use the existing feature branch for the current task. Do not create `feature/foo-complete` or `feature/foo-final`.
  * **Naming Convention:** Use descriptive names like `feature/task-description` or `fix/bug-id`. Avoid vague names like `temp`, `backup`, or `test`.
  * **Clean Up (Both Local and Remote):** Delete feature branches only AFTER they have been successfully merged and verified. **Mandated by Protocol 101 Part C.3**:
    ```bash
    # Verify the branch is merged first (e.g., via GitHub UI or git log)

    # Then delete (Mandatory two-step process)
    git branch -D feature/branch-name           # Delete local
    git push origin --delete feature/branch-name # Delete remote (Mandated by P101 v3.0)
    ```

## 2\. Destructive Commands (Protocol 101 Part B.1: Absolute Prohibition)

  * **NO Reckless Resets:** Never run `git reset --hard` unless:

    1.  You have explicitly verified that all valuable changes are pushed to a remote branch.
    2.  You have user permission.
    3.  If uncommitted work exists, run `git stash` first and inform the user.

    **Bad Example (Prohibited by Protocol 101):**

    ```bash
    git reset --hard origin/main  # ❌ Destroys uncommitted work
    ```

    **Good Example:**

    ```bash
    git stash                     # ✅ Save uncommitted work
    git reset --hard origin/main  # ✅ Reset with user permission
    git stash pop                 # ✅ Restore work (with user OK)
    ```

  * **NO Force Pushes:** Do not use `git push --force` or `-f` on shared branches.

  * **NO Rebase on Shared Branches:** Avoid `git rebase` on branches that others are working on (e.g., `main`, `dev`).

    **When Rebase IS Okay:**

      - Solo feature branches only
      - After `git fetch origin` to sync with remote
      - After reviewing potential conflicts

    **Example:**

    ```bash
    git fetch origin              # ✅ Sync first
    git rebase origin/main        # ✅ Only on solo feature branch
    ```

    **Interactive Rebase:**

      - If using `git rebase -i`, always review and test changes before pushing

## 3\. Workflow

  * **Protected Branches:** Assume `main` and `master` are protected. Always use a Pull Request workflow.
  * **Status Checks:** Always run `git status` before and after complex operations to confirm state.
  * **Context Awareness:** Before running a command, verify:
      * Which branch am I on?
      * What is the state of the remote? Run `git fetch origin` to update remote state before decisions.
      * Will this command destroy uncommitted or unpushed work?
      * Run `git diff --cached` (staged) and `git diff` (unstaged) to review changes before committing.
  * **PR Creation:** If pushing to `main` fails due to protections, create a PR:
    ```bash
    # Option 1: GitHub CLI (if installed)
    gh pr create --draft --title "WIP: Feature Name"

    # Option 2: Web UI
    # Navigate to GitHub and create PR manually
    ```

## 4\. Error Handling

  * **STOP and Analyze (Protocol 101 Part B.3: Prohibition of Sovereign Improvisation):** If a Git command fails, **STOP**. Do not blindly try a different command or create a new branch.
  * **Error Reporting Template:** When reporting errors to the user, use this format:
    ```
    ❌ Git Error Detected
    Error Message: [exact error output]
    Current Branch: [branch name]
    Git Status: [summary of git status]
    Suggested Safe Next Steps:
      1. [Option 1]
      2. [Option 2]
      3. [Option 3]
    ```
  * **Common Recoveries:**
      - **For merge conflicts:** Suggest `git mergetool` or abort with `git merge --abort`
      - **For rebase conflicts:** Suggest `git rebase --abort` or resolve and `git rebase --continue`
      - **For detached HEAD:** Suggest `git checkout <branch-name>` to return to a branch
  * **Ask for Guidance:** If the path forward involves potential data loss, ask the user for explicit permission.

## 5\. Sequential Workflow (The "Finish It" Rule)

  * **One Thing at a Time:** Do NOT start new work, create new files, or push new commits if a Pull Request is active and pending merge.
  * **Complete the Cycle:** The workflow is: `Feature Branch` -\> `PR` -\> `Merge` -\> `Cleanup (Delete Local AND Remote Branch)` -\> `Pull Main`. Only THEN can new work begin.
  * **Post-Merge Sync:** Immediately after a PR is merged, run:
    ```bash
    git checkout main
    git pull origin main
    ```
  * **No "Ride-Along" Commits:** Do not add unrelated changes (like documentation updates or rule files) to a feature branch that is already under review or ready for PR. Create a new branch *after* the current one is merged.
      * **Exception:** Urgent hotfixes may bypass this rule but MUST use a separate `hotfix/` branch and be merged via PR with expedited review. Follow the full cycle post-merge.

## 6\. Pre-Command Checklist

Before executing any Git command that modifies history or state, **output this checklist** and verify each item:

```
🔍 Pre-Command Checklist:
[ ] I am on the correct branch (run: git branch --show-current)
[ ] I have reviewed uncommitted changes (run: git status, git diff)
[ ] All valuable work is committed or stashed
[ ] I have user permission for destructive operations
[ ] I understand the consequences of this command
```

**For AI Assistants:** Mark each checkbox as you verify, then proceed only if all checks pass.

## 7\. Integration Notes

### For Antigravity

Reference this file in your agent prompts:

```
"Strictly adhere to .agent/git_safety_rules.md. Before any Git action, run the Pre-Command Checklist and confirm compliance."
```

### For Cursor

1.  **Option 1:** Add this file's content directly to Cursor's **Rules for AI** editor (Settings → Cursor Settings → Rules for AI)
2.  **Option 2:** Save as `.cursorrules` in your project root and reference it in prompts
3.  Use GitLens extension for visual verification of branch state

### For VS Code / Copilot

1.  Save this file as `.agent/git_safety_rules.md` or `.vscode/git_safety_rules.md`
2.  In Copilot settings, add a workspace prompt:
    ```
    "Follow the Git safety rules in .agent/git_safety_rules.md before executing any Git commands."
    ```
3.  Use extensions like **GitLens** for visual verification of branch state

### Testing

Simulate failure scenarios in a dummy repo to verify adherence:

  - Force a push failure to `main` and confirm no panic branching occurs
  - Trigger a merge conflict and verify proper error reporting
  - Test reset scenarios to ensure stash is used

### Enforcement Tools

  - **Git Hooks:** Use Husky to automate pre-commit/pre-push checks
  - **Security:** Pair with GitGuardian for secret scanning
  - **Monitoring:** Review LLM outputs regularly; if rules are violated, add negative examples to this file

## 8\. Security Considerations

  * **Secret Scanning:** Before committing, scan for secrets:
    ```bash
    git secrets --scan  # Or use GitGuardian
    ```
  * **AI Security Integration:** When suggesting code edits, incorporate security checks from established rulesets (e.g., OWASP, SecureCodeWarrior)
  * **Prompt Addition for AI:** "Before committing code changes, verify no hardcoded secrets, API keys, or credentials are present."

## 9\. CI/CD Integration

**GitHub Actions Workflows:** `.github/workflows/ci.yml`

Automated checks run on every push and pull request:

  - **Protocol 101 Verification (Functional Coherence)** - **VERIFIES ALL AUTOMATED TESTS PASS** (Supersedes Manifest Check)
  - **ShellCheck** - Lints shell scripts in `tools/`
  - **Python Linting** - Black formatting and Flake8 checks
  - **Testing** - Runs pytest for `council_orchestrator` and `mnemonic_cortex`
  - **Security Scanning** - Trivy vulnerability scanner

**Dependabot:** `.github/dependabot.yml`

Automated dependency updates:

  - GitHub Actions (weekly)
  - Python packages (weekly, with major version protections for torch/transformers)

-----

**Remember:** These rules exist to protect your work. When in doubt, ask the user before proceeding.

**Violations?** Update this file with the incident as a negative example to improve AI adherence over time.
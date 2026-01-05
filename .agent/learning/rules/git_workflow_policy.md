---
trigger: manual
---

## 🛠️ Project Sanctuary: Git Feature Workflow Rules (v3.0 - Strict Mode)

### 1. The Golden Rule: NO DIRECT WORK ON MAIN
*   **Check First**: Before writing a single line of code, run `git branch`.
*   **If on Main**: **STOP.** Create a feature branch immediately (`git checkout -b feat/your-task-name`).
*   **Violation**: Direct commits to `main` are strictly forbidden unless resolving a merge conflict during a pull.

### 2. Serial Execution (The "One at a Time" Rule)
*   **Focus**: You may only have **one** active feature branch at a time.
*   **No Hopping**: Do NOT create a new feature branch until the current one is:
    1.  **Pushed** to origin.
    2.  **Merged** by the User (PR complete).
    3.  **Deleted** locally and remotely.
*   **Why**: This prevents "context bleeding" and keeps the repository clean. Finish what you start.

### 3. The Lifecycle of a Feature
Every task MUST follow this exact cycle:

1.  **START**: `git checkout -b feat/description` (from fresh `main`).
2.  **WORK**: Edit files, run tests.
3.  **SAVE**: `git add .` -> `git commit -m "feat: description"`.
4.  **PUBLISH**: `git push origin feat/description`.
5.  **WAIT**: Ask User to review and merge the PR. **Do not touch the branch while waiting.**
6.  **SYNC**: `git checkout main` -> `git pull origin main`.
7.  **PRUNE**: `git branch -d feat/description` (Locally) + `git push origin --delete feat/description` (Remotely).

### 4. Integration & Safety
*   **Smart Commits**: Use `sanctuary-git-git-smart-commit` (or standard git) but ensure messages follow Conventional Commits (e.g., `feat:`, `fix:`, `docs:`).
*   **Status Checks**: Run `git status` frequently to ensure you are not committing unrelated files.
*   **Conflict Resolution**: If a conflict occurs, resolve it on the feature branch (merge `main` into `feat/...`), NOT on `main`.

### 5. Transition Rule
*   **Security Scan**: Check for open Dependabot alerts or PRs. If critical, prioritize them as the next task.
*   **Strategic Inquiry**: precise question: *"Branch merged and deleted. What is the next priority?"*.
*   **Zero Residue**: Ensure `git branch` shows only `main` before starting the next task.
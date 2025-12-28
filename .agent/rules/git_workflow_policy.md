---
trigger: always_on
---

## 🛠️ Project Sanctuary: Git Feature Workflow Rules (v2.0)

### 1. Feature Initialization (The "Start" Phase)

* **Intent Capture**: Verify the task details in the `TASKS/` directory before starting.
* **Mandatory Freshness**: Use `sanctuary-git-git-start-feature`. This tool now **automatically fetches** from `origin/main` to ensure your new branch is based on the most recent verified state.

* **Slug Identification**: Branch names are automatically generated as `feature/task-XXX-description` to maintain repo-wide consistency.

### 2. Iterative Development (The "Active" Phase)

* **Orchestrated Commits**: You may now pass a `files` list directly to `sanctuary-git-git-smart-commit`. This allows you to verify, stage, and commit in a single atomic operation, reducing "Staging Block" friction.

* 
**Context-Aware Safety**: Be aware that `smart_commit` (Protocol 101) is now intelligent: it will **skip strict code tests** for non-code artifacts like ADRs or Markdown documentation, while maintaining full enforcement for Python/Code files.

* **Synchronization Awareness**: Before pushing, use `sanctuary-git-git-get-status`. It now performs an async fetch to provide **"Honest Reporting"**—warning you if your local branch is behind the remote before you attempt a push.



### 3. Integration & Peer Review (The "Wait" Phase)

* **PR Handover**: Notify the user when technical objectives are met.
* **Execution Pause**: You **MUST wait** for the user to manually merge the PR. Do not modify the feature branch during this window to avoid merge conflicts.

* 
**Pre-Push Validation**: `sanctuary-git-git-push-feature` will now block and warn you if a rebase/pull is required to prevent "Push Failures".

### 4. Verification & Cleanup (The "Finish" Phase)

* **Remote Verification**: After the user confirms the merge, run `sanctuary-git-git-get-status`. This ensures your local view matches the remote state.

* **The "Fresh" Finish**: Use `sanctuary-git-git-finish-feature`. This tool now executes a **Mandatory Auto-Fetch** to verify the merge status against the fresh `origin/main` before allowing branch deletion.

* **Poka-Yoke Integrity**: If the finish tool detects uncommitted drift or a failed merge state, it will block deletion. Report this discrepancy to the user immediately.


### 5. Transition & Continuation (The "Next" Phase)

* **Strategic Inquiry**: Ask: *"The previous feature is sealed and cleaned. What is the next tactical priority?"*.
* **Task Selection**: Upon confirmation, immediately restart Step 1 for the next unit of work, leveraging the newly cleaned environment.
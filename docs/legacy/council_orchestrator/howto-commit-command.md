# How to Commit Using command.json (Protocol 101 v3.0 Compliant)

## Goal

Commit (and push) using `command.json` processed by the orchestrator while satisfying **Protocol 101 v3.0: Functional Coherence** (passing the automated test suite).

## TL;DR

Create a `command.json` with `git_operations` specifying the files to commit. The orchestrator will automatically execute the **Functional Coherence Test Suite**, and only if successful, will it stage the files, commit, and optionally push.

## Process Overview Diagram

The original manifest verification step is **permanently purged** and replaced by the mandatory test execution step.

```mermaid
flowchart TD
    A[Start: Files to commit] --> B[Create command.json with git_operations]
    B --> C[Run orchestrator]
    C --> D[Orchestrator executes Functional Coherence Test Suite]
    D --> E{Tests Pass?}
    E -->|Yes (P101 v3.0 Compliant)| F[Orchestrator executes git add/commit/push]
    E -->|No (Protocol Violation)| G[Commit Aborted - Fix Tests]
    F --> H[End: Files committed and pushed]
```

## 1\) Create command.json (Functional Integrity Mandate)

The orchestrator now mandates the successful execution of the automated test suite as the integrity check before any commit can be executed (Protocol 101 v3.0, Part A).

The orchestrator will now automatically:

  - **Run the Test Suite** (`./scripts/run_genome_tests.sh`).
  - If tests fail, the commit process **aborts**.
  - If tests succeed, it stages files and proceeds to commit.

## 2\) Run the Orchestrator

Execute the orchestrator to process the `command.json`. It will:

  - Execute the test suite for **Functional Coherence**.
  - Use `git rm` to stage deletions for files in `git_operations.files_to_remove`.
  - Run `git add` on all files.
  - Run `git commit` with your provided message.
  - Optionally `git push` if `push_to_origin` is true.

**Important:** Mechanical tasks are supported by the orchestrator schema under `git_operations` (add/remove/commit/push). The orchestrator ensures **Functional Coherence** is met before committing.

## Minimal Safe Command JSON

Include `output_artifact_path` and use `push_to_origin: false` for a dry run.

```json
{
  "task_description": "Commit orchestrator artifacts (dry-run)",
  "git_operations": {
    "files_to_add": [
      "council_orchestrator/command_git_ops.json",
      "../capture_code_snapshot.py"
    ],
    "files_to_remove": [
      "old_temp_file.txt"
    ],
    "commit_message": "orchestrator: add snapshot and command artifacts, remove temp file (dry-run)",
    "push_to_origin": false
  },
  "output_artifact_path": "council_orchestrator/command_results/commit_results.json",
  "config": {}
}
```

## Common Pitfalls and Troubleshooting (Reforged)

  - **Test Failure (Protocol Violation):** If the commit is aborted, it is a **Protocol 101 v3.0 Violation** due to a test failure. The only permitted action is to **STOP** and **REPORT THE FAILURE** to the Steward. You must fix the code or the test, ensuring all tests pass before re-running the command.
  - **Missing Files:** Ensure all files in `files_to_add` exist and are accessible by the orchestrator process.
  - **Timing Issues:** Minimize time between command creation and execution to reduce the race window where another process might introduce an un-tested file, potentially causing a test failure.

## Best Practices

  - Use `push_to_origin: false` for the first run to validate add+commit locally and confirm all tests pass before deploying.
  - Include `output_artifact_path` in your command.
  - Minimize time between command creation and execution.
  - Strictly adhere to **Part B: Action Integrity** by only using the whitelisted commands available through `git_operations`.

-----


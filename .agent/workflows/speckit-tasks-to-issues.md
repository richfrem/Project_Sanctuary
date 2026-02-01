---
description: Convert existing tasks into actionable, dependency-ordered GitHub issues for the feature based on available design artifacts.
tools: ['github/github-mcp-server/issue_write']
---


## Phase 0: Pre-Flight
```bash
python tools/cli.py workflow start --name speckit-tasks-to-issues --target "[Target]"
```
*This handles: Git state check, context alignment, spec/branch management.*

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. Run `scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").
1. From the executed script, extract the path to **tasks**.
1. Get the Git remote by running:

```bash
git config --get remote.origin.url
```

> [!CAUTION]
> ONLY PROCEED TO NEXT STEPS IF THE REMOTE IS A GITHUB URL

1. For each task in the list, use the GitHub MCP server to create a new issue in the repository that is representative of the Git remote.

> [!CAUTION]
> UNDER NO CIRCUMSTANCES EVER CREATE ISSUES IN REPOSITORIES THAT DO NOT MATCH THE REMOTE URL

---

## Universal Closure (MANDATORY)

After issue creation is complete, execute the standard closure sequence:

### Step A: Self-Retrospective
```bash
/workflow-retrospective
```
*Checks: Smoothness, gaps identified, Boy Scout improvements.*

### Step B: Workflow End
```bash
/workflow-end "chore: create GitHub issues for [FeatureName]" specs/[NNN]-[title]/
```
*Handles: Human review, git commit/push, PR verification, cleanup.*
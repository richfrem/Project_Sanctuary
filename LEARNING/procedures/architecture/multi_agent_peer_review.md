---
id: proc-multi-agent-peer-review
name: Multi-Agent Peer Review Dispatch
version: 1.0.0
domain: architecture
applicability: "Dispatching a bounded architectural or security review to an external CLI agent (Codex / Astra, Claude Code) under token-efficient parameters."
inputs:
  - backend: "Target CLI backend: codex, claude, agy"
  - model: "Target model identifier: gpt-6-astra, claude-3-5-sonnet, etc."
  - prompt_file: "Path to markdown prompt file"
  - output_file: "Path to destination review markdown file"
prerequisites:
  - "Target CLI agent binary is installed and authenticated (/opt/homebrew/bin/codex, claude)"
  - "Prompt file exists and contains clear context and objective questions"
steps:
  - "1. Verify binary availability and version"
  - "2. Prepare structured review prompt with source context"
  - "3. Execute run_agent.py via cli-agent harness"
  - "4. Inspect generated output file for substance and non-empty size"
success_criteria:
  - "Command exit status is 0"
  - "Output file exists and contains > 500 characters of actionable review"
abort_conditions:
  - "Backend CLI binary not found or authentication timed out"
  - "Subagent output is empty or truncated"
recovery_guidance:
  - "Verify network credentials or re-run with isolated prompt payload."
outcomes:
  environment: "macOS-darwin"
  success_count: 3
  failure_count: 0
  blocked_count: 0
  last_verified: "2026-11-02"
---

# Multi-Agent Peer Review Dispatch

## 1. Overview
Project Sanctuary operates as a multi-agent sovereign ecosystem. When complex architectural decisions, security assessments, or second opinions are required, agents dispatch bounded peer reviews through the `cli-agent` harness.

## 2. Execution Playbook

```bash
python3 /Users/richardfremmerlid/.gemini/antigravity-cli/skills/cli-agent/scripts/run_agent.py \
  /dev/null \
  <prompt_file> \
  <output_file> \
  "You are a peer sovereign engineering agent in Project Sanctuary. Review the provided prompt and deliver rigorous, actionable feedback." \
  --cli <backend> \
  --model <model>
```

## 3. Post-Condition Verification
Examine `<output_file>` to ensure the review completed successfully without simulated or synthetic claims.

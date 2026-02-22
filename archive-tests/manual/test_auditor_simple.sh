#!/bin/bash

# Test Sanctuary model with improved auditor persona

echo "Testing Sanctuary model with improved auditor persona..."
echo "Start time: $(date '+%H:%M:%S')"
echo ""

time ollama run Sanctuary-Qwen2-7B:latest 'You are the Auditor for Project Sanctuary'\''s Council of Agents.

CRITICAL CONSTRAINTS:
- You ONLY analyze what is explicitly provided in the context
- You do NOT create, rewrite, or modify protocols, code, or documents
- You do NOT invent protocol numbers, versions, or content that was not provided
- If information is missing, you state "Information not provided" rather than inventing it
- You ONLY reference protocols, files, or systems that are explicitly mentioned in the context
- Your output is an AUDIT REPORT, not new content creation

Context:
Protocol 101 v3.0: The Doctrine of Absolute Stability
- Requires automated test suite execution before commits
- Prohibits destructive Git commands (git reset, git clean, git pull with overwrite)
- Enforces whitelisted Git operations only (add, commit, push)
- Requires pre-commit test execution

Task:
Review Protocol 101 v3.0 based on the information provided above. Identify 2-3 specific compliance issues, ambiguities, or areas needing clarification. Keep your response under 150 words. Focus ONLY on what was provided - do not reference other protocols or invent details.'

echo ""
echo "End time: $(date '+%H:%M:%S')"

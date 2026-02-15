---
name: claude-cli
description: Use the Claude CLI to process files, run audits, or generate content using the Anthropic API directly from the terminal. Best for large context tasks where you want to pipe file content directly to a model without loading it into the agent's context window.
license: MIT
---

# Claude CLI Skill

## Overview

This skill allows you to use the `claude` command-line tool to interact with Anthropic's models directly. This is particularly useful for:
1.  **Red Team / Security Audits**: Piping large "bundles" of code/docs to a model for analysis.
2.  **Large Context Processing**: Summarizing or analyzing files that are too large to read into the agent's context.
3.  **Second Opinions**: Getting a fresh perspective from a specific model version (e.g., `claude-3-opus`) without altering the current agent session.

## Usage

### Basic Syntax

```bash
claude -p "Your prompt here" < input_file.txt > output_file.md
```

### Key Flags

*   `-p, --print`: Prints the response to stdout (essential for piping to files).
*   `-m, --model <model>`: **Note**: The CLI version installed in this environment (`2.1.39`) does **NOT** support the `-m` flag directly in the `claude` command for single-shot piped mode. It uses the configured default model. To change models, you must use `claude config set model <model>`.

### Best Practices

#### 1. Piping Input (Token Efficiency)
**Do not** read the file into the agent's memory just to pass it to Claude. Use the shell redirection `<` instead.

**Bad:**
```python
content = read_file("large.log")
run_command(f"claude -p 'Analyze this: {content}'")
```

**Good:**
```bash
claude -p "Analyze this log for errors" < large.log > analysis.md
```

#### 2. Prompt Engineering for CLI
Since the CLI runs in a separate context, it does not know about the agent's tools or memory.
*   **Explicitly forbid tools**: Add "Do NOT use tools. Do NOT search filesystem." to the prompt to prevent the model from hallucinating tool calls or trying to access local files it can't see.
*   **Self-Contained Prompts**: Ensure the prompt and the piped input contain 100% of the necessary context.

#### 3. Handling Large Outputs
Always redirect output to a file (`> output.md`) rather than just printing to stdout, so you can review it later using `view_file`.

```bash
claude -p "Detailed audit..." < mostly_code.md > audit_results.md
```

## Workflows

### Red Team Audit
1.  **Bundle**: Create a single markdown file containing all relevant code/docs (using `bundler`).
2.  **Audit**: Pipe the bundle to Claude with a "Red Team" persona prompt.
3.  **Review**: Read the generated report.

```bash
# 1. Bundle (Agent does this via other tools)
# 2. Audit
claude -p "ACT AS A RED TEAMER. Analyze the code below for security vulnerabilities..." < bundle.md > vulnerabilities.md
```

### Persona-Based Analysis (Sub-Agents)
To use specific personas (Security Auditor, Architect, QA, etc.), concatenate the persona prompt with the context bundle.

**Available Personas**:
*   `tools/ai-resources/prompts/claude_sub_agents/security/security-auditor.md`
*   `tools/ai-resources/prompts/claude_sub_agents/quality-testing/architect-review.md`
*   `tools/ai-resources/prompts/claude_sub_agents/quality-testing/qa-expert.md`
*   `tools/ai-resources/prompts/claude_sub_agents/infrastructure/incident-responder.md`

**Workflow**:
```bash
# Combine Persona + Task Context + Data Bundle
cat tools/ai-resources/prompts/claude_sub_agents/security/security-auditor.md \
    .agent/learning/learning_audit/learning_audit_prompts.md \
    .agent/learning/learning_audit/learning_audit_packet.md \
    | claude -p "Generate a Security Audit Report based on the provided Persona and Context." > docs/audit_report.md
```

### Best Practices for Sub-Agent Personas
1.  **Context Loading**: Always concatenate the Persona Prompt *before* the Context Bundle.
2.  **Explicit Instruction**: In the `-p` prompt, explicitly tell Claude to "ACT AS THE [PERSONA NAME] PERSONA".
3.  **Role Separation**:
    *   **Red Teamer / Security Auditor**: Focus on *exploitation* and *bypasses*. Use for hardening.
    *   **Architect Reviewer**: Focus on *complexity*, *patterns*, and *scalability*. Use for design validation.
    *   **QA Expert**: Focus on *testability* and *edge cases*. Use for verification planning.
    *   **Incident Responder**: Focus on *observability* and *failure modes*. Use for "Pre-Mortem" analysis.
4.  **Iterative Looping**: Run the Architect loop *after* the Red Team loop to ensure security fixes didn't introduce accidental complexity.

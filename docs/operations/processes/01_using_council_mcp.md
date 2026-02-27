# Tutorial: Using the Council Agent Plugin Integration

The **Council Agent Plugin Integration** is the orchestration engine of Project Sanctuary. It enables you to dispatch complex tasks to a group of specialized AI agents who deliberate, critique, and refine solutions before presenting them to you.

## What is the Council?

The Council is not a single agent; it's a **multi-agent system** that simulates a team of experts. When you send a task to the Council, it doesn't just answer; it:
1.  **Plans**: A Coordinator breaks down the task.
2.  **Executes**: Specialized agents (Strategist, Auditor, etc.) perform the work.
3.  **Reviews**: Agents critique each other's work.
4.  **Refines**: The solution is improved based on feedback.

## Basic Usage

To use the Council, you simply ask your LLM assistant to "dispatch" a task.

**Prompt:**
> "Ask the Council to review our current testing strategy."

Behind the scenes, the assistant calls the `council_dispatch` tool:

```python
council_dispatch(
    task_description="Review our current testing strategy",
    max_rounds=3
)
```

## Advanced Usage

### 1. Specific Agent Consultation

You can request a specific agent if you don't need the full council.

**Prompt:**
> "Have the Auditor check this code for security vulnerabilities."

**Tool Call:**
```python
council_dispatch(
    task_description="Check this code for security vulnerabilities",
    agent="auditor"
)
```

### 2. Output to File

You can have the Council write its final decision directly to a file.

**Prompt:**
> "Design a new API schema and save it to `docs/api/schema_v2.md`."

**Tool Call:**
```python
council_dispatch(
    task_description="Design a new API schema...",
    output_path="docs/api/schema_v2.md"
)
```

## Understanding the Agents

*   **Coordinator**: The project manager. Breaks down tasks, assigns work, and synthesizes results.
*   **Strategist**: The visionary. Focuses on long-term goals, alignment with principles, and high-level design.
*   **Auditor**: The critic. Checks for errors, security risks, and compliance with standards (like Protocol 101).

## Best Practices

*   **Be Specific**: The clearer your task description, the better the Council can plan.
*   **Use Context**: Provide relevant file paths or background info in your prompt.
*   **Iterate**: If the Council's first draft isn't perfect, ask follow-up questions or request a "second round" of deliberation.

## Next Steps

*   Learn about **[[README|Agent Personas]]** to create custom agents.
*   Explore **[[02_using_cortex_mcp|Cortex]]** to give the Council access to long-term memory.

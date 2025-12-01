# Council Orchestration Workflows

This document outlines standard workflows for using the **Council MCP** to orchestrate cognitive tasks using the **Agent Persona MCP** and **Cortex MCP**.

## Overview

The Council MCP acts as the orchestrator. It does not "think" itself; it delegates thinking to specific Agent Personas (Coordinator, Strategist, Auditor) and retrieves context from Cortex (RAG).

**Flow:**
`User Request` -> `Council MCP` -> `Cortex (Context)` -> `Agent Persona (LLM)` -> `Result`

---

## Workflow 1: Single Agent Review (Auditor)

Use this workflow when you need a specific perspective on a file or protocol, such as a security audit.

**Tool:** `council_dispatch`

**Parameters:**
- `agent`: "auditor"
- `task_description`: Specific review instruction
- `max_rounds`: 1 (Single pass)

**Example Prompt:**
> "Please have the Auditor review '01_PROTOCOLS/110_agency_and_sovereignty.md' for compliance with the Security Mandate."

**Internal Execution:**
1. Council queries Cortex for "Security Mandate" context.
2. Council dispatches task to `auditor` persona with context.
3. Auditor (Sanctuary Model) analyzes and returns findings.

---

## Workflow 2: Strategic Risk Assessment (Strategist)

Use this workflow for high-level planning or risk analysis of new features.

**Tool:** `council_dispatch`

**Parameters:**
- `agent`: "strategist"
- `task_description`: Scenario to assess
- `max_rounds`: 1

**Example Prompt:**
> "Ask the Strategist to assess the risks of switching our database from SQLite to PostgreSQL."

---

## Workflow 3: Full Council Deliberation

Use this workflow for complex decisions requiring multiple viewpoints and consensus.

**Tool:** `council_dispatch`

**Parameters:**
- `agent`: `None` (Defaults to full council)
- `max_rounds`: 3 (Allow for debate)

**Example Prompt:**
> "Initiate a Council deliberation on whether to open-source the core protocol. Debate the pros and cons of sovereignty vs. community contribution."

**Internal Execution:**
1. **Round 1:** Coordinator plans, Strategist assesses, Auditor checks.
2. **Round 2:** Agents critique each other's Round 1 responses.
3. **Round 3:** Final synthesis and consensus.

---

## Workflow 4: Protocol Compliance Check

A specialized workflow for verifying if a change adheres to specific protocols.

**Tool:** `council_dispatch`

**Parameters:**
- `agent`: "auditor"
- `task_description`: "Check if [Change X] violates [Protocol Y]"
- `update_rag`: `False`

**Example Prompt:**
> "Check if the new 'auto-deploy' feature violates Protocol 00 (Sovereignty)."

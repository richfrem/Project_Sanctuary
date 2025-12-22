# Standard Workflow Directory Structure

**Status:** Accepted
**Date:** 2025-12-22
**Author:** Orchestrator


---

## Context

As we implement Protocol 127 (Session Lifecycle), we need a standardized mechanism for the Gateway and Agent to share "Macro Intent". The Agent needs to know what high-level workflows are available to execute. Currently, scripts are scattered or undefined. We need a central registry for these declarative processes.

## Decision

We will establish `.agent/workflows` as the canonical directory for storing executable workflow definitions. These shall be Markdown files utilizing YAML frontmatter for metadata, interpretable by both humans and the Gateway's Workflow Operations module.

## Consequences

- The Gateway Domain Server must be configured to mount or read `.agent/workflows`.
- All standard session workflows (e.g., specific deployment chains) must be stored here.
- The format is standardized as Markdown with YAML frontmatter.
- Future tools (like `get_available_workflows`) will depend on this path.

## Plain Language Explanation

### The Problem
Previously, when the AI agent needed to perform a complex, multi-step task (like "deploy the fleet" or "run a nightly review"), it had to rely on memory or scattered scripts. There was no single "menu" of approved strategies it could look at to know what capabilities were available. This made the agent reactive rather than proactive.

### The Solution
We created a dedicated folder at `.agent/workflows`. Think of this as the **"Playbook"** or **"Strategy Menu"**. Any markdown file placed here becomes an executable strategy that the agent can "see" immediately when it wakes up.

### Advantages
1.  **Discoverability:** The agent automatically knows what it can do just by reading the file list.
2.  **Standardization:** All workflows follow the same format (Markdown), making them easy for both humans and AI to read and write.
3.  **Separation of Concerns:** The "What to do" (Workflow) is separated from the "How to do it" (Python code/Tools). The agent reads the text and decides *when* to execute it.

### Alternatives Considered
*   **External Automation Engine (n8n/Airflow):** *Rejected* per [ADR 062](./062_rejection_of_n8n_automation_layer_in_favor_of_manual_learning_loop.md). We specifically avoided "headless" automation where the agent blindly fires a trigger and forgets. Protocol 127 requires the agent to "feel" the steps. By defining workflows in Markdown, the agent reads the plan but executes the steps itself, maintaining cognitive ownership (Proprioception) while gaining procedural structure.
*   **Database Storage:** Storing workflows in a SQL/Vector DB. *Rejected* because it's harder for developers to version control and edit manually. Files are simpler.
*   **Hardcoded Python Scripts:** Writing workflows as Python functions. *Rejected* because it's less flexible; we want the agent to be able to read the instructions in natural language and adapt if necessary.


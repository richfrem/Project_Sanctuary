# Obsidian Agentic Capabilities & Requirements Summary

This document synthesizes the capabilities, specialized skills, and software requirements necessary for Project Sanctuary agents to fully leverage an Obsidian Vault, based on the Gemini 3.1 Pro Deep Research report.

## 1. Software Installation & Environment Requirements

To utilize the full suite of automation tools described in the research, the host environment must meet the following prerequisites:

*   **Obsidian Desktop Application**: Must be installed and running on the host machine. The official CLI communicates via an Inter-Process Communication (IPC) singleton lock, requiring an active instance.
*   **Obsidian CLI (v1.12+)**: Must be installed and available in the system PATH. Operates in "silent" mode by default.
*   **Agent Client Plugin (Optional but Recommended)**: A community plugin installed within the Obsidian vault to bridge agents using the Agent Client Protocol (ACP), allowing native `@notename` references and contextual awareness natively inside the vault UI.
*   **Target Plugins for Advanced Skills (Optional)**: If implementing visual boards or databases, the vault must have the relevant community plugins enabled (e.g., Dataview, Canvas, Obsidian Bases).

## 2. Command Line Capabilities (Obsidian CLI)

The CLI acts as the programmatic bridge for agents, bypassing the need for manual python text parsing where native Obsidian features are required.

*   `obsidian vault`: Retrieves telemetry (name, path, file count, size).
*   `obsidian daily`: Interacts with periodic daily notes for chronologs.
*   `obsidian read`: Resolves `[[wikilinks]]` internally without raw regex guessing.
*   `obsidian search --context`: Performs full-text search yielding JSON context blocks.
*   `obsidian properties`: Native parsing and writing of YAML frontmatter metadata.
*   `obsidian backlinks`: Instantly returns the graph network of files pointing to a target note.
*   **Developer Mode (`dev:eval`)**: Allows the agent to execute raw JavaScript directly against the Obsidian DOM and internal API.

## 3. Specialized Agent Skills

To move beyond generic Markdown generation, agents require distinct operational profiles (Skills) that master Obsidian's proprietary structures.

### Skill 1: Obsidian Markdown Mastery
*   **Capabilities**: Generating `[[Wikilinks]]`, `![[Embeds]]`, `> [!callout]` syntax, and Dataview-compliant YAML properties.
*   **Use Case**: Base requirement for note creation so the graph functions correctly.

### Skill 2: Obsidian Bases Manager
*   **Capabilities**: Interacting with `.base` files (YAML structures that render as databases, cards, and tables).
*   **Use Case**: Agents act as database administrators, updating multi-view layouts, defining formulas, and using global filters to triage tasks automatically.

### Skill 3: JSON Canvas Architect
*   **Capabilities**: Writing `.canvas` files (JSON Canvas Spec 1.0) defining text nodes, file nodes, and directional edges with specific Pixel coordinates.
*   **Use Case**: Agentic visual planning; the agent charts its execution path visually for human review rather than solely relying on text logs.

## 4. Architectural Impact on Work Packages

Based on this summary, **WP01 (Research Integration Strategy)** must be expanded. Instead of just deciding between "CLI vs Script," WP01 must produce a comprehensive **Plugin Architecture Plan**. This plan will define explicitly which of the above skills (CLI, Bases, Canvas, ACP) will be implemented as discrete Agent Skills within Project Sanctuary, ensuring our agents have the specific capabilities needed to master the vault.

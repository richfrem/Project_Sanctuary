# Architectural Synthesis: Kepano Obsidian Skills vs Project Sanctuary

**Date:** 2026-02-27
**Target:** `https://github.com/kepano/obsidian-skills`

## 1. Executive Summary
Kepano's Obsidian Skills repository provides excellent instructional prompts for how an LLM should parse rendering geometries (`.canvas`), data structures (`.base`), and linking syntax (wikilinks). However, architecturally, Kepano's implementation is entirely **prompt-based (Zero-Code)**, relying completely on Claude Code / OpenCode's native filesystem editing capabilities to interpret the instructions. 

In contrast, Project Sanctuary operates under the **Agent Skill Open Specifications** enforced by our `agent-scaffolders`. Sanctuary skills require rigorous Python scripts (`scripts/`), validation gates (`verify`), and programmatic guarantees to interact with the file system safely, rather than trusting the LLM to write raw text flawlessly.

## 2. Capabilities Mapped (The Kepano Blueprint)

While we reject Kepano's "prompt-only" execution model, the instructional blueprints within his `SKILL.md` files are highly valuable schemas that we will port into our Python execution logic:

### A. `json-canvas` (Canvas Geometry)
- **Insight:** Kepano explicitly defines the need for 16-character hex IDs, math spacing (50-100px apart), and structural rules (z-index via array order).
- **Sanctuary Transposition (WP07):** Our `json-canvas-architect` python skill will codify these rules, automatically enforcing ID generation and collision avoidance rather than relying on the LLM to do math.

### B. `obsidian-bases` (Data Views)
- **Insight:** Kepano provides a deep spec for `.base` YAML structures, specifically how formulas and filters are nested. He notes a critical pitfall: unescaped YAML quotes breaking the file.
- **Sanctuary Transposition (WP07):** Our `obsidian-bases-manager` python skill will use `ruamel.yaml` to guarantee lossless YAML parsing and safe injection of formulas, completely eliminating the LLM-syntax-error risk Kepano identifies.

### C. `obsidian-markdown` (Wikilinks & Embeds)
- **Insight:** Defines the strict difference between `[[wikilink]]` and `![[embed]]`, block references (`^block`), and callouts (`> [!note]`). 
- **Sanctuary Transposition (WP05):** We will build a Python `obsidian-parser` module that uses regex or markdown abstract syntax trees (AST) to reliably scrape and modify these constructs, ensuring the LLM doesn't accidentally break transclusions.

## 3. The `agent-skill-open-specifications` Boundary

As mandated by Project Sanctuary's `plugin-architecture`, our upcoming Obsidian skills will strictly adhere to the open specifications standard:
1. **`SKILL.md`**: Will contain the LLM persona, triggers, and instructions (partially adapted from Kepano's clear phrasing).
2. **`scripts/`**: Will contain the Python execution scripts (e.g., `read_canvas.py`, `write_base.py`, `parse_wikilinks.py`) that actually do the filesystem reading/writing. 
3. **Iron Core**: Agents will invoke the scripts rather than using native file editing tools to manipulate the complex `.canvas` and `.base` JSON/YAML trees.

## 4. Agnosticism vs Orchestration (The Guardian Boundary)

A critical architectural distinction must be maintained when building these new skills:
- **The Obsidian Skills MUST BE AGNOSTIC**: The `json-canvas-architect`, `obsidian-bases-manager`, and `obsidian-markdown` tools must know **nothing** about Project Sanctuary, the Constitution, the RLM, or Hugging Face. They are pure, generic tools that read and write Obsidian formats. They could be dropped into any other project and work perfectly.
- **The Guardian MUST BE CONTEXT-AWARE**: It is the `guardian-onboarding` skill (and the Orchestrator) that possesses the domain knowledge of Project Sanctuary. The Guardian will determine *what* Sanctuary data needs to be visualized, and it will invoke the generic Obsidian skills and pass them the payload to do the drawing.

*(Note: This mirrors the exact separation of concerns achieved in the `agent-loops` vs `orchestrator` refactor, where execution loops remain agnostic and the orchestrator handles the domain-specific routing).*

## 5. Conclusion
Kepano's repository is an excellent **Data Dictionary** for Obsidian formats, but it is not a structural blueprint for Project Sanctuary. We will extract the schemas and layout rules from his skills and embed them securely into our deterministically executing, completely agnostic Python ecosystem.

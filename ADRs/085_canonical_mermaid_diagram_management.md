# ADR 085: Canonical Mermaid Diagram Management

**Status:** Approved  
**Date:** 2025-12-31  
**Deciders:** Human Steward, Antigravity Agent  

---

## Context

Inline Mermaid diagrams (`\`\`\`mermaid` blocks) embedded directly in Markdown files cause **Mnemonic Bloat** when these files are captured in learning snapshots. A single inline diagram can be duplicated hundreds of times through recursive snapshot embedding (debrief → seal → audit), inflating file sizes from KB to GB.

## Decision

**No Direct Embedding of Mermaid Diagrams.**

All diagrams MUST follow the Canonical Diagram Pattern:

### The Pattern

1. **Create `.mmd` file** in `docs/architecture_diagrams/{category}/`:
   ```
   docs/architecture_diagrams/
   ├── rag/           # RAG architecture
   ├── system/        # Infrastructure/fleet
   ├── transport/     # MCP transport
   └── workflows/     # Process/workflow
   ```

2. **Add header metadata** to the `.mmd` file:
   ```text
   %% Name: My Diagram Title
   %% Description: What this diagram shows
   %% Location: docs/architecture_diagrams/category/my_diagram.mmd
   ```

3. **Generate PNG** using the render script:
   ```bash
   python3 scripts/render_diagrams.py docs/architecture_diagrams/category/my_diagram.mmd
   ```

4. **Reference in documents** with image AND source link:
   ```markdown
   ![Diagram Title](path/to/diagram.png)
   
   *Source: [diagram.mmd](path/to/diagram.mmd)*
   ```

## Consequences

### Positive
- **Snapshot Size Reduction**: Diagrams stored once, not duplicated per capture
- **Single Source of Truth**: One `.mmd` file per diagram, versioned in Git
- **Consistency**: All diagrams rendered with same tooling and styling
- **Auditability**: Source `.mmd` always linked for verification

### Negative
- **Extra Step**: Must run render script after diagram changes
- **Two Files**: `.mmd` source + `.png` output per diagram

## Compliance

Files with inline `\`\`\`mermaid` blocks MUST be refactored before inclusion in any manifest (`learning_manifest.json`, `learning_audit_manifest.json`, `red_team_manifest.json`).

Use this command to detect violations:
```bash
grep -rl '\`\`\`mermaid' . --include="*.md" | grep -v node_modules | grep -v .agent/learning/
```

---

*See also: [Task #154: Mermaid Rationalization](../tasks/todo/154_mermaid_rationalization.md)*

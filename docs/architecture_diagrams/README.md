# Architecture Diagrams

Central repository for all Project Sanctuary architecture diagrams.

## Directory Structure

```
docs/architecture_diagrams/
├── rag/                    # RAG architecture diagrams
├── system/                 # System/infrastructure diagrams  
├── transport/              # MCP transport diagrams
├── workflows/              # Process/workflow diagrams
└── README.md               # This file
```

## Diagram Inventory

| File | Description |
|------|-------------|
| **rag/** | |
| `basic_rag_architecture.mmd` | Ingestion and query pipelines for basic RAG |
| `advanced_rag_architecture.mmd` | MCP-enabled RAG with caching and routing |
| **system/** | |
| `sanctuary_mcp_overview.mmd` | High-level MCP ecosystem overview |
| `mcp_gateway_fleet.mmd` | Gateway-hosted fleet of 8 servers |
| `mcp_ecosystem_legacy_stdio.mmd` | Legacy 12-server stdio architecture |
| `council_orchestration_stack.mmd` | Council orchestration layers |
| `gateway_testing_architecture.mmd` | Testing infrastructure setup |
| **transport/** | |
| `mcp_sse_stdio_transport.mmd` | Dual transport architecture (SSE + STDIO) |
| `gateway_production_flow.mmd` | Production request flow |
| `mcp_testing_dev_paths.mmd` | Testing and development paths |
| **workflows/** | |
| `protocol_128_learning_loop.mmd` | Cognitive Continuity workflow |
| `llm_finetuning_pipeline.mmd` | Phoenix Forge fine-tuning pipeline |
| `mcp_request_validation_flow.mmd` | Request validation middleware flow |

## Regenerating Images

When diagrams are updated, regenerate the PNG images:

```bash
# Render all diagrams to PNG
python3 scripts/render_diagrams.py

# Render specific diagram
python3 scripts/render_diagrams.py my_diagram.mmd  

# Render as SVG instead
python3 scripts/render_diagrams.py --svg

# Check which images are outdated
python3 scripts/render_diagrams.py --check
```

### Prerequisites

The render script uses [mermaid-cli](https://github.com/mermaid-js/mermaid-cli). It will be automatically installed via `npx` on first run.

## Adding New Diagrams

1. Create a new `.mmd` file in the appropriate subfolder
2. Add a header with name/description:
   ```text
   %% Name: My New Diagram
   %% Description: What this diagram shows
   %% Location: docs/architecture_diagrams/folder/my_diagram.mmd
   
   flowchart TB
       ...
   ```
3. Run `python3 scripts/render_diagrams.py` to generate the image
4. Reference the image in docs: `![Diagram](#)`

## Single Source of Truth (SSOT)

All Mermaid diagrams MUST be maintained here per **[ADR 085: Canonical Mermaid Diagram Management](../../ADRs/085_canonical_mermaid_diagram_management.md)**.

Inline `\`\`\`mermaid` blocks in other docs are **prohibited**. Replace with image references pointing to these canonical sources. This reduces "Mnemonic Bloat" in snapshots and ensures consistency.

### Compliance Check
```bash
grep -rl '\`\`\`mermaid' . --include="*.md" | grep -v node_modules | grep -v .agent/learning/
```

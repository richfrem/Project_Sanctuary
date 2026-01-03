# Scripts Directory

Utility scripts for Project Sanctuary operations, verification, and maintenance.

## Quick Reference

### Core CLIs

| Script | Purpose | Usage |
|--------|---------|-------|
| `cortex_cli.py` | Main Cortex operations CLI | `python scripts/cortex_cli.py --help` |
| `cortex_cli.py guardian` | Guardian mode awakening (preferred) | `python scripts/cortex_cli.py guardian` |
| `domain_cli.py` | Domain-specific operations | `python scripts/domain_cli.py --help` |
| `init_session.py` | Session initialization | `python scripts/init_session.py` |
| `guardian_wakeup.py` | ⚠️ Legacy - use `cortex_cli.py guardian` | `python scripts/guardian_wakeup.py` |

### Diagram & Documentation

| Script | Purpose |
|--------|---------|
| `render_diagrams.py` | Render `.mmd` → `.png` |
| `mermaid_inventory.py` | Scan/audit Mermaid diagrams |
| `replace_mermaid_with_images.py` | Replace inline Mermaid with images |
| `extract_orphaned_diagrams.py` | Find unreferenced diagrams |

### Configuration & Generation

| Script | Purpose |
|--------|---------|
| `generate_gateway_config.py` | Generate MCP gateway configuration |
| `generate_mcp_config.py` | Generate MCP server configs |
| `generate_learning_snapshot.py` | Create learning snapshots |
| `create_fastmcp_server.py` | Scaffold new MCP server |

### Verification & Testing

| Script | Purpose |
|--------|---------|
| `verify_full_fleet.py` | Check all 12 MCP servers health |
| `verify_rag_capacities.py` | Validate RAG pipeline |
| `verify_domain_routing.py` | Test domain routing |
| `verify_cortex_image_content.py` | Check Cortex image integrity |
| `security_scan.py` | Run security checks |
| `run_integration_tests.sh` | Run integration test suite |

### Hugging Face

| Script | Purpose |
|--------|---------|
| `hf_upload_assets.py` | Upload assets to HF |
| `hf_decorate_readme.py` | Update HF README cards |

### Maintenance

| Script | Purpose |
|--------|---------|
| `update_genome.sh` | Update cognitive genome snapshot |
| `fix_hardcoded_paths.py` | Fix hardcoded path references |
| `ensure_manifest_completeness.py` | Validate manifest integrity |
| `validate_manifest.py` | Check manifest structure |
| `capture_code_snapshot.py` | Create code snapshots |

### Specialized

| Script | Purpose |
|--------|---------|
| `glyph_forge.py` | Generate code glyphs |
| `stress_test_adr084.py` | ADR 084 entropy testing |
| `manual_test_deliberation.py` | Manual Council deliberation test |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `stabilizers/` | Vector DB consistency tools |
| `archive/` | Deprecated scripts |

## Common Patterns

```bash
# Check fleet health
python scripts/verify_full_fleet.py

# Render all diagrams
python scripts/render_diagrams.py

# Initialize session
python scripts/init_session.py

# Guardian wakeup (Protocol 128 bootloader)
python scripts/cortex_cli.py guardian

# Run security scan
python scripts/security_scan.py
```

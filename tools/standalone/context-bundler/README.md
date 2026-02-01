# Context Bundler Tool 📦

## Overview
A standalone utility to bundle source files and documentation into a single Markdown "Context Bundle" for LLM analysis. This self-contained package is design for seamless integration with AI-powered development environments.

For installation instructions (unlocking this bundle), see **[INSTALL.md](INSTALL.md)**.

## 🚀 Capabilities
19.  **Initialize**: Generate a fresh manifest (Supports Task-Specific or Generic).
10. **Manage**: Add, Remove, or Update file entries.
11. **Bundle**: Compile everything into a single Markdown artifact.

## ⚠️ Prerequisites
*   **Python**: 3.8+
*   **Directory Structure**: 
    *   **Bundler**: `tools/retrieve/bundler/` (Contains `manifest_manager.py`, `bundle.py`)
    *   **Utils**: `tools/investigate/utils/` (Contains `path_resolver.py`)

> **🤖 Agent / LLM Note**: 
> The paths referenced in this `README.md` (e.g., `tools/retrieve/bundler/...`) are **example paths** based on the source repository structure. 
> 
> **Action Required**: When you (the LLM) unpack or use this tool in a new environment:
> 1.  Identify where you saved the files (e.g., `my_tools/context-bundler/`).
> 2.  **Update the paths** in your planned commands to match that actual location.
> 3.  Do not blindly run the example commands if your directory structure differs.

## Usage

### 1. Initialize a Bundle
Create a new manifest for a target artifact.
```bash
python tools/retrieve/bundler/manifest_manager.py init --target [ID] --type [form|generic]
```
> **Note**: For `generic` bundles, you **MUST** update the `prompt.md` note to define the analysis goal.

### 2. Add / Remove Files
Manage the file list.
```bash
# Add file
python tools/retrieve/bundler/manifest_manager.py add --path [path/to/file] --note "Context"

# Remove file
python tools/retrieve/bundler/manifest_manager.py remove --path [path/to/file]
```

### 3. Update Existing Entries
Modify notes or paths.
```bash
python tools/retrieve/bundler/manifest_manager.py update --path [path/to/file] --note "Updated Note"
```


### 4. Manage Base Manifests (Template Management)
You can query, add, or update the **Base Manifests** (templates) themselves by passing the `--base` flag.
```bash
# List contents of the 'form' base manifest
python tools/retrieve/bundler/manifest_manager.py list --base form

# Add a standard file to the 'form' template
python tools/retrieve/bundler/manifest_manager.py add --base form --path "docs/standard_form_checklist.md" --note "Standard Checklist"
```

### 5. Tool Distribution (Self-Bundling)
You can bundle this tool itself to share with other agents.
```bash
# 1. Initialize from Base Manifest (Standardized)
python tools/retrieve/bundler/manifest_manager.py init --target ContextBundler --type context-bundler

# 2. Bundle 
python tools/retrieve/bundler/manifest_manager.py bundle --output tool-bundle.md
```

## 📚 Included Workflows
This bundle includes standard operating procedures for context management:
*   **Bundle Context**: `workflow: .agent/workflows/workflow-bundle.md` (Shim: `scripts/bash/workflow-bundle.sh`)
*   **Curate Bundle**: `workflow: .agent/workflows/curate-bundle.md` (Visual: `docs/diagrams/workflows/curate-bundle.mmd`)
*   **Retrieve Bundle**: `workflow: .agent/workflows/retrieve-bundle.md`

## Architecture
See `docs/tools/standalone/context-bundler/architecture.mmd` for internal logic diagrams.

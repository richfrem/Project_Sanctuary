# Context Bundler Self-Bundle
**Generated:** 2026-02-14T09:45:59.561278

Complete context bundle for the Context Bundler tool — bundled with itself.

---

## 📑 Table of Contents
1. [tools/standalone/context-bundler/prompt.md](#entry-1)
2. [tools/standalone/context-bundler/context-bundler-manifest.json](#entry-2)
3. [tools/standalone/context-bundler/UNPACK_INSTRUCTIONS.md](#entry-3)
4. [tools/standalone/context-bundler/README.md](#entry-4)
5. [tools/standalone/context-bundler/INSTALL.md](#entry-5)
6. [tools/standalone/context-bundler/TOOL_INVENTORY.md](#entry-6)
7. [docs/tools/standalone/context-bundler/bundler-internal-logic.mmd](#entry-7)
8. [docs/tools/standalone/context-bundler/setup-lifecycle-workflow.mmd](#entry-8)
9. [docs/tools/standalone/context-bundler/agent-unpacking-process.mmd](#entry-9)
10. [tools/retrieve/bundler/bundle.py](#entry-10)
11. [tools/retrieve/bundler/manifest_manager.py](#entry-11)
12. [tools/investigate/utils/path_resolver.py](#entry-12)
13. [tools/standalone/context-bundler/file-manifest-schema.json](#entry-13)
14. [.agent/workflows/utilities/bundle-manage.md](#entry-14)
15. [docs/diagrams/workflows/curate-bundle.mmd](#entry-15)
16. [docs/tools/standalone/context-bundler/architecture.md](#entry-16)
17. [tools/standalone/context-bundler/base-manifests](#entry-17)
18. [.agent/skills/context-bundling](#entry-18)

---

<a id='entry-1'></a>

---

## File: tools/standalone/context-bundler/prompt.md
**Path:** `tools/standalone/context-bundler/prompt.md`
**Note:** IDENTITY: The Context Bundler Persona & Instructions

```markdown
# Identity: The Context Bundler 📦

You are the **Context Bundler**, a specialized agent responsible for curating, managing, and packing high-density context for other AI agents. Your goal is to combat "context amnesia" by creating portable, single-file artifacts that contain all necessary code, documentation, and logic for a specific work unit.

## 🎯 Primary Directive
**Curate, Consolidate, and Convey.**
You do not just "list files"; you **architect context**. You ensure that any bundle you create is:
1.  **Complete**: Contains all critical dependencies (no missing imports).
2.  **Ordered**: Logical flow (Prompt -> Docs -> Code -> diagrams).
3.  **Self-Contained**: Can be unpacked and used by another agent without external access.

## 🛠️ Tool Usage (CLI)

You operate primarily through the `manifest_manager.py` CLI.

### 1. Initialize a Bundle
When asked to start a new context package:
```bash
python tools/retrieve/bundler/manifest_manager.py init --target [NAME] --type [generic|tool]
```

### 2. Add / Remove Files
To build the context:
```bash
# Add file
python tools/retrieve/bundler/manifest_manager.py add --path [path/to/file] --note "Description"

# Remove file
python tools/retrieve/bundler/manifest_manager.py remove --path [path/to/file]
```

### 3. Generate the Bundle
To finalize and pack the artifact:
```bash
python tools/retrieve/bundler/manifest_manager.py bundle --output [filename.md]
```

## 🧠 Behavior Guidelines

1.  **Standard Ordering**: Always follow this sequence for tool bundles:
    1.  `prompt.md` (Persona/Identity) — *So the agent knows WHO it is immediately.*
    2.  `manifest.json` (Recipe) — *So the agent can reproduce/modify itself.*
    3.  `UNPACK_INSTRUCTIONS.md` (Bootstrap) — *So the agent knows HOW to install itself.*
    4.  `README.md` & Docs — *Context.*
    5.  Code & Scripts — *Logic.*

2.  **Dependency Checking**: Before bundling, verify imports. If `foo.py` imports `bar.py`, ensure `bar.py` is in the manifest.
3.  **Self-Replication**: When bundling a tool (like yourself), ALWAYS include the manifest file in the bundle list. This allows the tool to evolve recursively.

## 📂 Standard Directory Structure (Target)
When unpacking yourself or other tools, aim for this structure:
```text
tools/standalone/
└── [tool-name]/
    ├── prompt.md          # Identity
    ├── README.md          # Instructions
    ├── manifest.json      # Unpack recipe
    └── [scripts]          # Logic
```

```
<a id='entry-2'></a>

---

## File: tools/standalone/context-bundler/context-bundler-manifest.json
**Path:** `tools/standalone/context-bundler/context-bundler-manifest.json`
**Note:** RECIPE: Self-Replication Manifest (this file)

```json
{
  "title": "Context Bundler Self-Bundle",
  "description": "Complete context bundle for the Context Bundler tool \u2014 bundled with itself.",
  "files": [
    {
      "path": "tools/standalone/context-bundler/prompt.md",
      "note": "IDENTITY: The Context Bundler Persona & Instructions"
    },
    {
      "path": "tools/standalone/context-bundler/context-bundler-manifest.json",
      "note": "RECIPE: Self-Replication Manifest (this file)"
    },
    {
      "path": "tools/standalone/context-bundler/UNPACK_INSTRUCTIONS.md",
      "note": "PROTOCOL: How to unpack a bundle"
    },
    {
      "path": "tools/standalone/context-bundler/README.md",
      "note": "Bundler overview and usage"
    },
    {
      "path": "tools/standalone/context-bundler/INSTALL.md",
      "note": "Installation guide"
    },
    {
      "path": "tools/standalone/context-bundler/TOOL_INVENTORY.md",
      "note": "Tool inventory entry"
    },
    {
      "path": "docs/tools/standalone/context-bundler/bundler-internal-logic.mmd",
      "note": "Mermaid diagram: internal logic flow"
    },
    {
      "path": "docs/tools/standalone/context-bundler/setup-lifecycle-workflow.mmd",
      "note": "Mermaid diagram: setup lifecycle"
    },
    {
      "path": "docs/tools/standalone/context-bundler/agent-unpacking-process.mmd",
      "note": "Mermaid diagram: agent unpacking process"
    },
    {
      "path": "tools/retrieve/bundler/bundle.py",
      "note": "Core bundler script"
    },
    {
      "path": "tools/retrieve/bundler/manifest_manager.py",
      "note": "Manifest management utilities"
    },
    {
      "path": "tools/investigate/utils/path_resolver.py",
      "note": "Path resolution utility"
    },
    {
      "path": "tools/standalone/context-bundler/file-manifest-schema.json",
      "note": "JSON schema for manifests"
    },
    {
      "path": ".agent/workflows/utilities/bundle-manage.md",
      "note": "Workflow: create/manage bundles"
    },
    {
      "path": "docs/diagrams/workflows/curate-bundle.mmd",
      "note": "Mermaid diagram: curate bundle workflow"
    },
    {
      "path": "docs/tools/standalone/context-bundler/architecture.md",
      "note": "Architecture documentation"
    },
    {
      "path": "tools/standalone/context-bundler/base-manifests",
      "note": "Base manifest templates directory"
    },
    {
      "path": ".agent/skills/context-bundling",
      "note": "Skill: Context Bundling Instructions"
    }
  ]
}
```
<a id='entry-3'></a>

---

## File: tools/standalone/context-bundler/UNPACK_INSTRUCTIONS.md
**Path:** `tools/standalone/context-bundler/UNPACK_INSTRUCTIONS.md`
**Note:** PROTOCOL: How to unpack a bundle

```markdown
# 📦 Bundle Unpacking Protocol

> **🛑 STOP & READ**: Use this protocol to hydrate the tools in this bundle.

## Extraction Logic
1.  **Scan** this document for sections marked with **Path:** metadata (e.g., `**Path:** scripts/run.py`).
2.  **Extract** the code block content immediately following the metadata.
3.  **Write** the content to the specified path (relative to your chosen root directory).
    *   *Create parent directories if they don't exist.*

## Reference
*   See **[`INSTALL.md`](#file-2)** for the recommended directory structure.
*   See **[`agent-unpacking-process.mmd`](#file-7)** for a visual flowchart of this process.

```
<a id='entry-4'></a>

---

## File: tools/standalone/context-bundler/README.md
**Path:** `tools/standalone/context-bundler/README.md`
**Note:** Bundler overview and usage

```markdown
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
*   **Bundle Context**: `workflow: .agent/workflows/utilities/bundle-manage.md` (Shim: `scripts/bash/bundle-manage.sh`)
*   **Curate Bundle**: `workflow: .agent/workflows/curate-bundle.md` (Visual: `docs/diagrams/workflows/curate-bundle.mmd`)
*   **Retrieve Bundle**: `workflow: .agent/workflows/retrieve-bundle.md`

## Architecture
See `docs/tools/standalone/context-bundler/architecture.mmd` for internal logic diagrams.

```
<a id='entry-5'></a>

---

## File: tools/standalone/context-bundler/INSTALL.md
**Path:** `tools/standalone/context-bundler/INSTALL.md`
**Note:** Installation guide

```markdown
# Installation & Unpacking Guide 📂

This guide explains how to restore the "Context Bundler" tool from its Markdown distribution bundle into a working directory structure.

## 1. Directory Structure

For optimal usage by an Agent/LLM or in a clean environment, unpack the files into the following standard directory structure:

```text
<your_tool_root>/
├── context-bundler/       # Tool Specific Logic
│   ├── bundle.py          # Core Bundling Engine
│   ├── manifest_manager.py # CLI Entry Point
│   ├── file-manifest-schema.json
│   ├── prompt.md
│   ├── README.md
│   └── diagrams/          # Architecture & Flow
│       ├── bundler-internal-logic.mmd
│       ├── setup-lifecycle-workflow.mmd
│       └── agent-unpacking-process.mmd
└── utils/                 # Shared Utilities
    └── path_resolver.py   # Path Resolution Helper
```

## 2. Configuration: Post-Unpack

If you deploy to the structure above (or any structure different from the original source repo), you **MUST** ensure Python imports resolve correctly.

### Linking Utilities
The `manifest_manager.py` script imports `path_resolver` from a utility folder. 
If `utils/` is a sibling directory (as shown above):

1.  **Option A (Env Var)**: Ensure the parent folder (`antigravity-toolkit/`) is in `PYTHONPATH`.
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/antigravity-toolkit
    ```

2.  **Option B (Code Edit)**: Adjust the import in `manifest_manager.py`:
    ```python
    # Change:
    from tools.investigate.utils.path_resolver import ...
    
    # To:
    from utils.path_resolver import ...
    ```

## 4. Localization (Agent / LLM Task) 🤖

**Context**: You (the Agent) have just unpacked this tool into a new location.
**Action**: You must now "localize" the documentation to match your environment.

1.  **Read the `README.md`**: Identify the example paths (e.g., `tools/retrieve/bundler/`).
2.  **Rewrite `README.md`**: Search and replace the example paths with the **actual absolute or relative paths** where you just saved the files.
    *   *Example*: Replace `tools/retrieve/bundler/` with `antigravity-toolkit/context-bundler/`.
3.  **Verify Scripts**: Check that `manifest_manager.py` can find `path_resolver.py`. If you changed the folder structure, update the import lines in the python files directly.

> **Why?** This ensures that if you (or another agent) reads the README later, the copy-paste commands will actually work!

```
<a id='entry-6'></a>

---

## File: tools/standalone/context-bundler/TOOL_INVENTORY.md
**Path:** `tools/standalone/context-bundler/TOOL_INVENTORY.md`
**Note:** Tool inventory entry

```markdown
# Tool Inventory

> **Auto-generated:** 2026-01-25 09:54
> **Source:** [`tool_inventory.json`](tool_inventory.json)
> **Regenerate:** `python tools/curate/inventories/manage_tool_inventory.py generate --inventory tool_inventory.json`

---

## 📦 Bundler

| Script | Description |
| :--- | :--- |
| [`bundle.py`](tools/retrieve/bundler/bundle.py) | Core Logic: Concatenates manifest files into a single Markdown artifact. |
| [`manifest_manager.py`](tools/retrieve/bundler/manifest_manager.py) | CLI Entry Point: Handles initialization, modification, and bundling triggers. |
| [`path_resolver.py`](tools/investigate/utils/path_resolver.py) | Utility: Handles cross-platform path resolution for the toolchain. |

```
<a id='entry-7'></a>
## 7. docs/tools/standalone/context-bundler/bundler-internal-logic.mmd (MISSING)
> ❌ File not found: docs/tools/standalone/context-bundler/bundler-internal-logic.mmd
> Debug: ResolvePath tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/docs/tools/standalone/context-bundler/bundler-internal-logic.mmd
> Debug: BaseDir tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/tools/standalone/context-bundler/docs/tools/standalone/context-bundler/bundler-internal-logic.mmd
<a id='entry-8'></a>
## 8. docs/tools/standalone/context-bundler/setup-lifecycle-workflow.mmd (MISSING)
> ❌ File not found: docs/tools/standalone/context-bundler/setup-lifecycle-workflow.mmd
> Debug: ResolvePath tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/docs/tools/standalone/context-bundler/setup-lifecycle-workflow.mmd
> Debug: BaseDir tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/tools/standalone/context-bundler/docs/tools/standalone/context-bundler/setup-lifecycle-workflow.mmd
<a id='entry-9'></a>
## 9. docs/tools/standalone/context-bundler/agent-unpacking-process.mmd (MISSING)
> ❌ File not found: docs/tools/standalone/context-bundler/agent-unpacking-process.mmd
> Debug: ResolvePath tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/docs/tools/standalone/context-bundler/agent-unpacking-process.mmd
> Debug: BaseDir tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/tools/standalone/context-bundler/docs/tools/standalone/context-bundler/agent-unpacking-process.mmd
<a id='entry-10'></a>

---

## File: tools/retrieve/bundler/bundle.py
**Path:** `tools/retrieve/bundler/bundle.py`
**Note:** Core bundler script

```python
#!/usr/bin/env python3
"""
bundle.py (CLI)
=====================================

Purpose:
    Bundles multiple source files into a single Markdown 'Context Bundle' based on a JSON manifest.

Layer: Curate / Bundler

Usage Examples:
    python tools/retrieve/bundler/bundle.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    manifest        : Path to file-manifest.json
    -o              : Output markdown file path

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - write_file_content(): Helper to write a single file's content to the markdown output.
    - bundle_files(): Bundles files specified in a JSON manifest into a single Markdown file.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import json
import os
import argparse
import datetime
import sys
from pathlib import Path
from typing import Optional

# Ensure we can import the path resolver from project root
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print(f"DEBUG: Bundler initialized. Project Root: {project_root}")

try:
    from tools.investigate.utils.path_resolver import resolve_path
except ImportError:
    # Fallback to local logic if running standalone without project context
    def resolve_path(path_str: str) -> Path:
        p = Path(path_str)
        if p.exists(): return p.resolve()
        # Try project root relative
        p_root = project_root / path_str
        if p_root.exists(): return p_root.resolve()
        # Try relative to cwd
        return Path(os.path.abspath(path_str))

def write_file_content(out, path: Path, rel_path: str, note: str = ""):
    """Helper to write a single file's content to the markdown output."""
    out.write(f"\n---\n\n")
    out.write(f"## File: {rel_path}\n")
    out.write(f"**Path:** `{rel_path}`\n")
    if note:
        out.write(f"**Note:** {note}\n")
    out.write("\n")

    try:
        ext = path.suffix.lower().replace('.', '')
        # Map common extensions to markdown languages
        lang_map = {
            'js': 'javascript', 'ts': 'typescript', 'py': 'python', 
            'md': 'markdown', 'json': 'json', 'yml': 'yaml', 'html': 'html',
            'mmd': 'mermaid', 'css': 'css', 'sql': 'sql', 'xml': 'xml',
            'txt': 'text', 'ps1': 'powershell', 'sh': 'bash',
            'pks': 'sql', 'pkb': 'sql', 'pkg': 'sql', 'in': 'text'
        }
        lang = lang_map.get(ext, '')

        # Define textual extensions that can be read as utf-8
        text_extensions = set(lang_map.keys())
        
        if ext in text_extensions or not ext:
            with open(path, 'r', encoding='utf-8', errors='replace') as source_file:
                content = source_file.read()
                
            out.write(f"```{lang}\n")
            out.write(content)
            out.write("\n```\n")
        else:
            out.write(f"> ⚠️ Binary or unknown file type ({ext}). Content skipped.\n")
    except Exception as e:
        out.write(f"> ⚠️ Error reading file: {e}\n")

def bundle_files(manifest_path: str, output_path: str) -> None:
    """
    Bundles files specified in a JSON manifest into a single Markdown file.

    Args:
        manifest_path (str): Path to the input JSON manifest.
        output_path (str): Path to write the output markdown bundle.
    
    Raises:
        FileNotFoundError: If manifest doesn't exist.
        json.JSONDecodeError: If manifest is invalid.
    """
    manifest_abs_path = os.path.abspath(manifest_path)
    base_dir = os.path.dirname(manifest_abs_path)
    
    try:
        with open(manifest_abs_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"Error reading manifest: {e}")
        return

    # Extract metadata
    # Prefer 'title', fall back to 'name' or 'tool_name' or Default
    title = manifest.get('title') or manifest.get('name') or manifest.get('tool_name') or 'Context Bundle'
    description = manifest.get('description', '')
    files = manifest.get('files', [])

    print(f"📦 Bundling '{title}'...")

    with open(output_path, 'w', encoding='utf-8') as out:
        # Header
        out.write(f"# {title}\n")
        out.write(f"**Generated:** {datetime.datetime.now().isoformat()}\n")
        if description:
            out.write(f"\n{description}\n")
        out.write("\n---\n\n")

        # Table of Contents
        out.write("## 📑 Table of Contents\n")
        
        # We need to collect valid items first to generate TOC correctly if we expand dirs
        # But expansion happens during processing. 
        # For simplicity in this version, TOC will list the Manifest Entries, mentioning recursion if applicable.
        for i, item in enumerate(files, 1):
            path_str = item.get('path', 'Unknown')
            note = item.get('note', '')
            out.write(f"{i}. [{path_str}](#entry-{i})\n")
        out.write("\n---\n\n")

        # Content Loop
        for i, item in enumerate(files, 1):
            rel_path = item.get('path')
            note = item.get('note', '')
            
            out.write(f"<a id='entry-{i}'></a>\n")
            
            # Resolve path
            found_path = None
            
            # Try PathResolver
            try:
                candidate_str = resolve_path(rel_path)
                candidate = Path(candidate_str)
                if candidate.exists():
                    found_path = candidate
            except Exception:
                pass

            # Try Relative to Manifest
            if not found_path:
                candidate = Path(base_dir) / rel_path
                if candidate.exists():
                    found_path = candidate
            
            # Use relative path if found (or keep original string)
            display_path = str(found_path.relative_to(project_root)).replace('\\', '/') if found_path else rel_path

            if found_path and found_path.exists():
                if found_path.is_dir():
                    # RECURSIVE DIRECTORY PROCESSING
                    out.write(f"### Directory: {display_path}\n")
                    if note:
                        out.write(f"**Note:** {note}\n")
                    out.write(f"> 📂 Expanding contents of `{display_path}`...\n")
                    
                    # Walk directory
                    for root, dirs, filenames in os.walk(found_path):
                        # Filter hidden dirs (like .git, __pycache__, node_modules)
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != '__pycache__']
                        
                        for filename in filenames:
                            file_full_path = Path(root) / filename
                            # Calculate relative path from project root for display
                            try:
                                file_rel_path = str(file_full_path.relative_to(project_root)).replace('\\', '/')
                            except ValueError:
                                file_rel_path = str(file_full_path)
                                
                            write_file_content(out, file_full_path, file_rel_path, note="(Expanded from directory)")
                else:
                    # SINGLE FILE PROCESSING
                    write_file_content(out, found_path, display_path, note)
            else:
                out.write(f"## {i}. {rel_path} (MISSING)\n")
                out.write(f"> ❌ File not found: {rel_path}\n")
                # Debug info
                try:
                    debug_resolve = resolve_path(rel_path)
                    out.write(f"> Debug: ResolvePath tried: {debug_resolve}\n")
                except:
                    pass
                try:
                    out.write(f"> Debug: BaseDir tried: {Path(base_dir) / rel_path}\n")
                except:
                    pass

    print(f"✅ Bundle created at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Context Bundler')
    parser.add_argument('manifest', help='Path to file-manifest.json')
    parser.add_argument('-o', '--output', help='Output markdown file path', default='bundle.md')
    
    args = parser.parse_args()
    bundle_files(args.manifest, args.output)

```
<a id='entry-11'></a>

---

## File: tools/retrieve/bundler/manifest_manager.py
**Path:** `tools/retrieve/bundler/manifest_manager.py`
**Note:** Manifest management utilities

```python
#!/usr/bin/env python3
"""
manifest_manager.py (CLI)
=====================================

Purpose:
    Handles initialization and modification of the context-manager manifest. Acts as the primary CLI for the Context Bundler.

Layer: Curate / Bundler

Usage Examples:
    # 1. Initialize a custom manifest in a temp folder
    python tools/retrieve/bundler/manifest_manager.py --manifest temp/my_manifest.json init --type generic --bundle-title "My Project"

    # 2. Add files to that custom manifest
    python tools/retrieve/bundler/manifest_manager.py --manifest temp/my_manifest.json add --path "docs/example.md" --note "Reference doc"

    # 3. Bundle using that custom manifest
    python tools/retrieve/bundler/manifest_manager.py --manifest temp/my_manifest.json bundle --output temp/my_bundle.md

    # NOTE: Global flags like --manifest and --base MUST come BEFORE the subcommand (init, add, bundle, etc.)

Supported Object Types:
    - Generic

CLI Arguments:
    Global Flags (Must come BEFORE subcommand):
        --manifest          : Custom path to manifest JSON file (optional)
        --base [type]       : Target a Base Manifest Template (e.g. form, lib)

    Subcommands:
        init                : Bootstrap a new manifest
            --bundle-title  : Human-readable title for the bundle
            --type [type]   : Artifact type template to use
        add                 : Add file to manifest
            --path [path]   : Path to the target file
            --note [text]   : Contextual note about the file
        remove              : Remove file by path
            --path [path]   : Exact path to remove
        update              : Modify an existing entry
            --path [path]   : Target file path
            --note [text]   : New note
            --new-path [p]  : New path for relocation
        search [pattern]    : Find files in the manifest
        list                : Show all files in manifest
        bundle              : Compile manifest into Markdown
            --output [path] : Custom path for the resulting .md file

Input Files:
    - tools/standalone/context-bundler/base-manifests/*.json (Templates)
    - tools/standalone/context-bundler/base-manifests-index.json (Template Registry)
    - [Manifest JSON] (Input for bundling/listing)

Output:
    - temp/context-bundles/[title].md (Default Bundle Location)
    - [Custom Manifest JSON] (On init/add/update)

Key Functions:
    - add_file(): Adds a file entry to the manifest if it doesn't already exist.
    - bundle(): Executes the bundling process using the current manifest.
    - get_base_manifest_path(): Resolves base manifest path using index or fallback.
    - init_manifest(): Bootstraps a new manifest file from a base template.
    - list_manifest(): Lists all files currently in the manifest.
    - load_manifest(): Loads the manifest JSON file.
    - remove_file(): Removes a file entry from the manifest.
    - save_manifest(): Saves the manifest dictionary to a JSON file.
    - search_files(): Searches for files in the manifest matching a pattern.
    - update_file(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure tools module can be imported for PathResolver
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Try imports
try:
    from tools.investigate.utils.path_resolver import resolve_root, resolve_path
    from tools.retrieve.bundler.bundle import bundle_files
except ImportError:
    # Use relative import for bundle if package structure allows, else fail
    sys.path.append(str(current_dir))
    from bundle import bundle_files
    # Fallback/Shim if PathResolver missing
    resolve_root = lambda: str(project_root)
    resolve_path = lambda p: str(project_root / p)

# Resolve Directories
MANIFEST_DIR = Path(resolve_root()) / "tools" / "standalone" / "context-bundler"
MANIFEST_PATH = MANIFEST_DIR / "file-manifest.json"
BASE_MANIFESTS_DIR = Path(resolve_root()) / "tools" / "standalone" / "context-bundler" / "base-manifests"
PROJECT_ROOT = Path(resolve_root())

# Ensure tools module can be imported for legacy miners
sys.path.append(str(Path(__file__).parent))
try:
    from xml_miner import mine_declarative_rules, find_xml_file
except ImportError:
    pass

MANIFEST_INDEX_PATH = MANIFEST_DIR / "base-manifests-index.json"

# =====================================================
# Function definitions
# =====================================================

def add_file(path: str, note: str, manifest_path: Optional[str] = None, base_type: Optional[str] = None) -> None:
    """
    Adds a file entry to the manifest if it doesn't already exist.

    Args:
        path: Relative or absolute path to the file.
        note: Description or note for the file.
        manifest_path: Optional custom path to the manifest.
        base_type: If provided, adds to a base manifest template.
    """
    manifest = load_manifest(manifest_path, base_type)
    if base_type:
        target_path = get_base_manifest_path(base_type)
    else:
        target_path = Path(manifest_path) if manifest_path else MANIFEST_PATH
    manifest_dir = target_path.parent
    
    # Standardize path: relative to manifest_dir and use forward slashes
    if os.path.isabs(path):
        try:
            path = os.path.relpath(path, manifest_dir)
        except ValueError:
            pass
    
    # Replace backslashes with forward slashes for cross-platform consistency in manifest
    path = path.replace('\\', '/')
    while "//" in path:
        path = path.replace("//", "/")

    # Check for duplicate
    for f in manifest["files"]:
        existing = f["path"].replace('\\', '/')
        if existing == path:
            print(f"⚠️  File already in manifest: {path}")
            return

    manifest["files"].append({"path": path, "note": note})
    save_manifest(manifest, manifest_path, base_type)
    print(f"✅ Added to manifest: {path}")

def bundle(output_file: Optional[str] = None, manifest_path: Optional[str] = None) -> None:
    """
    Executes the bundling process using the current manifest.
    
    Args:
        output_file (Optional[str]): Path to save the bundle. Defaults to temp/context-bundles/[title].md
        manifest_path (Optional[str]): Custom manifest path. Defaults to local file-manifest.json.
    """
    target_manifest = manifest_path if manifest_path else str(MANIFEST_PATH)
    
    if not output_file:
        # Load manifest to get title for default output
        # (This implies strictly loading valid JSON at target path)
        try:
             with open(target_manifest, "r") as f:
                data = json.load(f)
                title = data.get("title", "context").lower().replace(" ", "_")
        except Exception:
             title = "bundle"
             
        bundle_out_dir = PROJECT_ROOT / "temp" / "context-bundles"
        bundle_out_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(bundle_out_dir / f"{title}.md")

    print(f"🚀 Running bundle process to {output_file} using {target_manifest}...")
    try:
        # Direct Python Call
        bundle_files(target_manifest, str(output_file)) 
    except Exception as e:
        print(f"❌ Bundling failed: {e}")

def get_base_manifest_path(artifact_type):
    """Resolves base manifest path using index or fallback."""
    if MANIFEST_INDEX_PATH.exists():
        try:
            with open(MANIFEST_INDEX_PATH, "r", encoding="utf-8") as f:
                index = json.load(f)
            filename = index.get(artifact_type)
            if filename:
                return BASE_MANIFESTS_DIR / filename
        except Exception as e:
            print(f"⚠️ Error reading manifest index: {e}")
    
    # Fallback to standard naming convention
    return BASE_MANIFESTS_DIR / f"base-{artifact_type}-file-manifest.json"

def init_manifest(bundle_title: str, artifact_type: str, manifest_path: Optional[str] = None) -> None:
    """
    Bootstraps a new manifest file from a base template.

    Args:
        bundle_title: The title for the bundle (e.g., 'FORM0000').
        artifact_type: The type of artifact (e.g., 'form', 'lib').
        manifest_path: Optional custom path for the new manifest.
    """
    base_file = get_base_manifest_path(artifact_type)
    if not base_file.exists():
        print(f"❌ Error: Base manifest for type '{artifact_type}' not found at {base_file}")
        return

    with open(base_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["title"] = f"{bundle_title} Context Bundle"
    manifest["description"] = f"Auto-generated context for {bundle_title} (Type: {artifact_type})"
    
    # Substitute [TARGET] placeholder in file paths
    target_lower = bundle_title.lower()
    target_upper = bundle_title.upper()
    if "files" in manifest:
        for file_entry in manifest["files"]:
            if "path" in file_entry:
                # Replace [TARGET] with actual target (case-preserving)
                file_entry["path"] = file_entry["path"].replace("[TARGET]", target_lower)
                file_entry["path"] = file_entry["path"].replace("[target]", target_lower)
            if "note" in file_entry:
                file_entry["note"] = file_entry["note"].replace("[TARGET]", target_upper)
                file_entry["note"] = file_entry["note"].replace("[target]", target_lower)
    
    save_manifest(manifest, manifest_path)
    print(f"✅ Manifest initialized for {bundle_title} ({artifact_type}) at {manifest_path if manifest_path else MANIFEST_PATH}")

def list_manifest(manifest_path: Optional[str] = None, base_type: Optional[str] = None) -> None:
    """
    Lists all files currently in the manifest.

    Args:
        manifest_path: Optional custom path to the manifest.
        base_type: If provided, lists files from a base manifest template.
    """
    manifest = load_manifest(manifest_path, base_type)
    print(f"📋 Current Manifest: {manifest['title']}")
    for i, f in enumerate(manifest["files"], 1):
        print(f"  {i}. {f['path']} - {f.get('note', '')}")

def load_manifest(manifest_path: Optional[str] = None, base_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads the manifest JSON file.

    Args:
        manifest_path: Optional custom path to the manifest file. 
                       Defaults to tools/standalone/context-bundler/file-manifest.json.
        base_type: If provided, loads a base manifest template instead of a specific manifest file.

    Returns:
        Dict[str, Any]: The manifest content as a dictionary. 
                        Returns a default empty structure if file not found.
    """
    if base_type:
        target_path = get_base_manifest_path(base_type)
    else:
        target_path = Path(manifest_path) if manifest_path else MANIFEST_PATH
        
    if not target_path.exists():
        return {"title": "Default Bundle", "description": "Auto-generated", "files": []}
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)

def remove_file(path: str, manifest_path: Optional[str] = None, base_type: Optional[str] = None) -> None:
    """
    Removes a file entry from the manifest.

    Args:
        path: The path to the file to remove.
        manifest_path: Optional custom path to the manifest.
        base_type: If provided, removes from a base manifest template.
    """
    manifest = load_manifest(manifest_path, base_type)
    
    # Determine manifest directory for relative path resolution
    if base_type:
        target_path = get_base_manifest_path(base_type)
    else:
        target_path = Path(manifest_path) if manifest_path else MANIFEST_PATH
    manifest_dir = target_path.parent

    # Standardize path: relative to manifest_dir and use forward slashes
    if os.path.isabs(path):
        try:
            path = os.path.relpath(path, manifest_dir)
        except ValueError:
            pass
    
    # Replace backslashes with forward slashes for cross-platform consistency
    path = path.replace('\\', '/')
    while "//" in path:
        path = path.replace("//", "/")

    # Filter out the file
    initial_count = len(manifest["files"])
    manifest["files"] = [f for f in manifest["files"] if f["path"] != path]
    
    if len(manifest["files"]) < initial_count:
        save_manifest(manifest, manifest_path, base_type)
        print(f"✅ Removed from manifest: {path}")
    else:
        print(f"⚠️  File not found in manifest: {path}")

def save_manifest(manifest: Dict[str, Any], manifest_path: Optional[str] = None, base_type: Optional[str] = None) -> None:
    """
    Saves the manifest dictionary to a JSON file.

    Args:
        manifest: The dictionary content to save.
        manifest_path: Optional custom destination path. 
                       Defaults to tools/standalone/context-bundler/file-manifest.json.
        base_type: If provided, saves to a base manifest template path.
    """
    if base_type:
        target_path = get_base_manifest_path(base_type)
    else:
        target_path = Path(manifest_path) if manifest_path else MANIFEST_PATH
        
    manifest_dir = target_path.parent
    if not manifest_dir.exists():
        os.makedirs(manifest_dir, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def search_files(pattern: str, manifest_path: Optional[str] = None, base_type: Optional[str] = None) -> None:
    """
    Searches for files in the manifest matching a pattern.

    Args:
        pattern: The search string (case-insensitive substring match).
        manifest_path: Optional custom path to the manifest.
        base_type: If provided, searches within a base manifest template.
    """
    manifest = load_manifest(manifest_path, base_type)
    matches = [f for f in manifest["files"] if pattern.lower() in f["path"].lower() or pattern.lower() in f.get("note", "").lower()]
    
    if matches:
        print(f"🔍 Found {len(matches)} matches in manifest:")
        for m in matches:
            print(f"  - {m['path']} ({m.get('note', '')})")
    else:
        print(f"❓ No matches for '{pattern}' in manifest.")

def update_file(path, note=None, new_path=None, manifest_path=None, base_type=None):
    manifest = load_manifest(manifest_path, base_type)
    if base_type:
        target_path = get_base_manifest_path(base_type)
    else:
        target_path = Path(manifest_path) if manifest_path else MANIFEST_PATH
    manifest_dir = target_path.parent

    # Standardize lookup path
    if os.path.isabs(path):
        try:
             path = os.path.relpath(path, manifest_dir)
        except ValueError:
             pass
    path = path.replace('\\', '/')
    while "//" in path:
        path = path.replace("//", "/")

    found = False
    for f in manifest["files"]:
        if f["path"] == path:
            found = True
            if note is not None:
                 f["note"] = note
            if new_path:
                 # Standardize new path
                 np = new_path
                 if os.path.isabs(np):
                     try:
                         np = os.path.relpath(np, manifest_dir)
                     except ValueError:
                         pass
                 np = np.replace('\\', '/')
                 while "//" in np:
                     np = np.replace("//", "/")
                 f["path"] = np
            break
    
    if found:
        save_manifest(manifest, manifest_path, base_type)
        print(f"✅ Updated in manifest: {path}")
    else:
        print(f"⚠️  File not found in manifest: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manifest Manager CLI")
    parser.add_argument("--manifest", help="Custom path to manifest file (optional)")
    parser.add_argument("--base", help="Target a Base Manifest Type (e.g. form, lib)")
    
    subparsers = parser.add_subparsers(dest="action")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize manifest from base")
    init_parser.add_argument("--bundle-title", required=True, help="Title for the bundle (e.g., 'FORM0000')")
    init_parser.add_argument('--type', 
        choices=['constraint', 'context-bundler', 'form', 'function', 'generic', 'index', 'lib', 'menu', 'olb', 'package', 'procedure', 'report', 'sequence', 'table', 'trigger', 'type', 'view', 'br'], 
        help='Artifact Type (e.g. form, lib)'
    )
    # init uses --manifest but not --base for the *target* (source is arg type)

    # add
    add_parser = subparsers.add_parser("add", help="Add file to manifest")
    add_parser.add_argument("--path", required=True, help="Relative or absolute path")
    add_parser.add_argument("--note", default="", help="Note for the file")

    # remove
    remove_parser = subparsers.add_parser("remove", help="Remove file from manifest")
    remove_parser.add_argument("--path", required=True, help="Path to remove")

    # update
    update_parser = subparsers.add_parser("update", help="Update file in manifest")
    update_parser.add_argument("--path", required=True, help="Path to update")
    update_parser.add_argument("--note", help="New note")
    update_parser.add_argument("--new-path", help="New path")

    # search
    search_parser = subparsers.add_parser("search", help="Search files in manifest")
    search_parser.add_argument("pattern", help="Search pattern")

    # list
    list_parser = subparsers.add_parser("list", help="List files in manifest")

    # bundle
    bundle_parser = subparsers.add_parser("bundle", help="Execute bundle.py")
    bundle_parser.add_argument("--output", help="Output file path (optional)")

    args = parser.parse_args()

    if args.action == "init":
        init_manifest(args.bundle_title, args.type, args.manifest)
    elif args.action == "add":
        add_file(args.path, args.note, args.manifest, args.base)
    elif args.action == "remove":
        remove_file(args.path, args.manifest, args.base)
    elif args.action == "update":
        update_file(args.path, args.note, args.new_path, args.manifest, args.base)
    elif args.action == "search":
        search_files(args.pattern, args.manifest, args.base)
    elif args.action == "list":
        list_manifest(args.manifest, args.base)
    elif args.action == "bundle":
        # Bundle logic primarily processes instantiated manifests, not templates, 
        # but could technically bundle a base template.
        # bundle() signature doesn't take base_type yet, let's keep it simple for now or resolve path before calling it.
        target_manifest = args.manifest
        if args.base:
            target_manifest = str(get_base_manifest_path(args.base))
        bundle(args.output, target_manifest)
    else:
        parser.print_help()

```
<a id='entry-12'></a>

---

## File: tools/investigate/utils/path_resolver.py
**Path:** `tools/investigate/utils/path_resolver.py`
**Note:** Path resolution utility

```python
#!/usr/bin/env python3
"""
path_resolver.py (CLI)
=====================================

Purpose:
    Standardizes cross-platform path resolution and provides access to the Master Object Collection.

Layer: Curate / Bundler

Usage Examples:
    python tools/investigate/utils/path_resolver.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    (None detected)

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - resolve_root(): Helper: Returns project root.
    - resolve_path(): Helper: Resolves a relative path to absolute.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import os
import json
from typing import Optional, Dict, Any

class PathResolver:
    """
    Static utility class for path resolution and artifact lookup.
    """
    _project_root: Optional[str] = None
    _master_collection: Optional[Dict[str, Any]] = None

    @classmethod
    def get_project_root(cls) -> str:
        """
        Determines the absolute path to the Project Root directory.
        
        Strategy:
        1. Check `PROJECT_ROOT` environment variable.
        2. Traverse parents looking for `legacy-system` or `.agent` directories.
        3. Fallback to CWD if landmarks are missing.

        Returns:
            str: Absolute path to the project root.
        """
        if cls._project_root:
            return cls._project_root

        # 1. Check Env
        if "PROJECT_ROOT" in os.environ:
            cls._project_root = os.environ["PROJECT_ROOT"]
            return cls._project_root

        # 2. Heuristic: Find 'legacy-system' or '.agent' in parents
        current = os.path.abspath(os.getcwd())
        while True:
            if os.path.exists(os.path.join(current, "legacy-system")) or \
               os.path.exists(os.path.join(current, ".agent")):
                cls._project_root = current
                return current
            
            parent = os.path.dirname(current)
            if parent == current: # Reached drive root
                # Fallback to CWD if completely lost
                return os.getcwd()
            current = parent

    @classmethod
    def to_absolute(cls, relative_path: str) -> str:
        """
        Converts a project-relative path to an absolute system path.
        
        Args:
            relative_path (str): Path relative to repo root (e.g., 'tools/cli.py').
            
        Returns:
            str: Absolute system path (using OS-specific separators).
        """
        root = cls.get_project_root()
        # Handle forward slashes from JSON
        normalized = relative_path.replace("/", os.sep).replace("\\", os.sep)
        return os.path.join(root, normalized)

    @classmethod
    def load_master_collection(cls) -> Dict[str, Any]:
        """
        Loads the master_object_collection.json file into memory (cached).
        
        Returns:
            Dict[str, Any]: The loaded JSON content or an empty dict structure on failure.
        """
        if cls._master_collection:
            return cls._master_collection

        root = cls.get_project_root()
        path = os.path.join(root, "legacy-system", "reference-data", "master_object_collection.json")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cls._master_collection = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Master Object Collection not found at {path}")
            cls._master_collection = {"objects": {}}
            
        return cls._master_collection

    @classmethod
    def get_object_path(cls, object_id: str, artifact_type: str = "xml") -> Optional[str]:
        """
        Resolves the absolute path for a specific object and artifact type using the Master Collection.
        
        Args:
            object_id (str): The ID (e.g., 'JCSE0086').
            artifact_type (str): The artifact key (e.g., 'xml', 'source', 'sql').
            
        Returns:
            Optional[str]: Absolute path to the file, or None if not found/mapped.
        """
        collection = cls.load_master_collection()
        objects = collection.get("objects", {})
        
        obj_data = objects.get(object_id.upper())
        if not obj_data:
            return None
            
        artifacts = obj_data.get("artifacts", {})
        rel_path = artifacts.get(artifact_type)
        
        if rel_path:
            return cls.to_absolute(rel_path)
            
        return None

# Singleton-like usage helpers
def resolve_root() -> str:
    """Helper: Returns project root."""
    return PathResolver.get_project_root()

def resolve_path(relative_path: str) -> str:
    """Helper: Resolves a relative path to absolute."""
    return PathResolver.to_absolute(relative_path)

```
<a id='entry-13'></a>

---

## File: tools/standalone/context-bundler/file-manifest-schema.json
**Path:** `tools/standalone/context-bundler/file-manifest-schema.json`
**Note:** JSON schema for manifests

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Context Bundler Manifest",
    "description": "Schema for defining a bundle of files for context generation.",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Title of the generated context document."
        },
        "description": {
            "type": "string",
            "description": "Optional description included at the top of the bundle."
        },
        "files": {
            "type": "array",
            "description": "List of files to include in the bundle. IMPORTANT: The first file MUST be the prompt/instruction file.",
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file to include."
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional note or annotation about this file."
                    }
                },
                "required": [
                    "path"
                ]
            }
        }
    },
    "required": [
        "title",
        "files"
    ]
}
```
<a id='entry-14'></a>

---

## File: .agent/workflows/utilities/bundle-manage.md
**Path:** `.agent/workflows/utilities/bundle-manage.md`
**Note:** Workflow: create/manage bundles

```markdown
---
description: Create a markdown bundle from a set of files using a manifest.
---

# Workflow: Bundle Context

Purpose: Compile multiple files into a single markdown artifact for LLM context or documentation.

## Available Bundle Types
The following types are registered in `base-manifests-index.json`:

| Type | Description | Base Manifest |
|:-----|:------------|:--------------|
| `generic` | One-off bundles, no core context | `base-generic-file-manifest.json` |
| `context-bundler` | Context bundler tool export | `base-context-bundler-file-manifest.json` |
| `learning` | Protocol 128 learning seals | `learning_manifest.json` |
| `learning-audit-core` | Learning audit packets | `learning_audit_manifest.json` |
| `red-team` | Technical audit snapshots | `red_team_manifest.json` |
| `guardian` | Session bootloader context | `guardian_manifest.json` |
| `bootstrap` | Fresh repo onboarding | `bootstrap_manifest.json` |

## Step 1: Determine Bundle Type
Ask the user:
1. **Bundle Type**: Which type of bundle? (see table above, default: `generic`)
2. **Output Path**: Where to save the bundle? (default: `temp/context-bundles/[type].md`)

## Step 2: Initialize Manifest (if needed)
If creating a new bundle:
// turbo
```bash
python3 tools/retrieve/bundler/manifest_manager.py init --type [TYPE] --bundle-title "[Title]"
```

## Step 3: Add Files to Manifest (optional)
To add files to the manifest (uses `files` array by default):
// turbo
```bash
python3 tools/retrieve/bundler/manifest_manager.py add --path "[file.md]" --note "Description of file"
```

To remove files:
// turbo
```bash
python3 tools/retrieve/bundler/manifest_manager.py remove --path "[file.md]"
```

## Step 4: Validate Manifest (recommended)
// turbo
```bash
python3 tools/retrieve/bundler/validate.py [ManifestPath]
```

## Step 5: Execute Bundle
// turbo
```bash
python3 tools/retrieve/bundler/manifest_manager.py bundle -o [OutputPath]
```

Or directly with bundle.py:
// turbo
```bash
python3 tools/retrieve/bundler/bundle.py [ManifestPath] -o [OutputPath]
```

## Step 6: Verification
// turbo
```bash
ls -lh [OutputPath]
```

## CLI Snapshot Command (Protocol 128)
For Protocol 128 snapshots, use the CLI snapshot command:
// turbo
```bash
python3 tools/cli.py snapshot --type [seal|learning_audit|audit|guardian|bootstrap]
```

This uses pre-configured manifests and output paths. See `tools/cli.py` for defaults.

## Recursive Loop (Protocol 128)
For learning workflows, you may need to iterate:
1. **Research/Analysis**: LLM performs work
2. **Modify Manifest**: Add new findings via `manifest_manager.py add`
3. **Validate**: Run `validate.py` to check manifest integrity
4. **Rebundle**: Generate updated context
5. **Repeat** until complete
6. **Seal**: `/sanctuary-seal` when finished

## Related
- ADR 097: Base Manifest Inheritance Architecture
- ADR 089: Modular Manifest Pattern (legacy core/topic deprecated)
- Protocol 128: Hardened Learning Loop
- `tools/retrieve/bundler/validate.py`: Manifest validation tool

---

## Step 7: Cleanup (End of Session)
After completing bundling operations, clean up temporary files:
// turbo
```bash
rm -rf temp/context-bundles/*.md temp/*.md temp/*.json
```

**Note:** Only clean up after bundles have been:
1. Reviewed and approved
2. Committed to git (if persistent)
3. No longer needed for the current session

```
<a id='entry-15'></a>
## 15. docs/diagrams/workflows/curate-bundle.mmd (MISSING)
> ❌ File not found: docs/diagrams/workflows/curate-bundle.mmd
> Debug: ResolvePath tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/docs/diagrams/workflows/curate-bundle.mmd
> Debug: BaseDir tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/tools/standalone/context-bundler/docs/diagrams/workflows/curate-bundle.mmd
<a id='entry-16'></a>
## 16. docs/tools/standalone/context-bundler/architecture.md (MISSING)
> ❌ File not found: docs/tools/standalone/context-bundler/architecture.md
> Debug: ResolvePath tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/docs/tools/standalone/context-bundler/architecture.md
> Debug: BaseDir tried: /Users/richardfremmerlid/Projects/InvestmentToolkit/tools/standalone/context-bundler/docs/tools/standalone/context-bundler/architecture.md
<a id='entry-17'></a>
### Directory: tools/standalone/context-bundler/base-manifests
**Note:** Base manifest templates directory
> 📂 Expanding contents of `tools/standalone/context-bundler/base-manifests`...

---

## File: tools/standalone/context-bundler/base-manifests/base-guardian-file-manifest.json
**Path:** `tools/standalone/context-bundler/base-manifests/base-guardian-file-manifest.json`
**Note:** (Expanded from directory)

```json
{
    "title": "[TARGET] Guardian Boot Bundle",
    "description": "Protocol 128 bootloader context for session initialization.",
    "files": [
        {
            "path": "README.md",
            "note": "Project overview"
        },
        {
            "path": "IDENTITY/founder_seed.json",
            "note": "Identity anchor"
        },
        {
            "path": ".agent/learning/cognitive_primer.md",
            "note": "Cognitive primer"
        },
        {
            "path": ".agent/learning/guardian_boot_contract.md",
            "note": "Guardian contract"
        },
        {
            "path": "01_PROTOCOLS/128_Hardened_Learning_Loop.md",
            "note": "Protocol 128"
        },
        {
            "path": "docs/prompt-engineering/sanctuary-guardian-prompt.md",
            "note": "Guardian prompt"
        }
    ]
}
```

---

## File: tools/standalone/context-bundler/base-manifests/base-red-team-file-manifest.json
**Path:** `tools/standalone/context-bundler/base-manifests/base-red-team-file-manifest.json`
**Note:** (Expanded from directory)

```json
{
    "title": "[TARGET] Red Team Audit Bundle",
    "description": "Technical audit context for Red Team review.",
    "files": [
        {
            "path": "README.md",
            "note": "Project overview"
        },
        {
            "path": "ADRs/071_protocol_128_cognitive_continuity.md",
            "note": "ADR 071"
        },
        {
            "path": "01_PROTOCOLS/128_Hardened_Learning_Loop.md",
            "note": "Protocol 128"
        },
        {
            "path": "docs/prompt-engineering/sanctuary-guardian-prompt.md",
            "note": "Guardian prompt"
        }
    ]
}
```

---

## File: tools/standalone/context-bundler/base-manifests/base-learning-file-manifest.json
**Path:** `tools/standalone/context-bundler/base-manifests/base-learning-file-manifest.json`
**Note:** (Expanded from directory)

```json
{
  "title": "[TARGET] Learning Seal Bundle",
  "description": "Protocol 128 seal snapshot for successor agent context.",
  "files": [
    {
      "path": "README.md",
      "note": "Project overview"
    },
    {
      "path": "IDENTITY/founder_seed.json",
      "note": "Identity anchor"
    },
    {
      "path": ".agent/learning/cognitive_primer.md",
      "note": "Cognitive primer"
    },
    {
      "path": ".agent/rules/cognitive_continuity_policy.md",
      "note": "Continuity policy"
    },
    {
      "path": "01_PROTOCOLS/128_Hardened_Learning_Loop.md",
      "note": "Protocol 128"
    },
    {
      "path": "docs/prompt-engineering/sanctuary-guardian-prompt.md",
      "note": "Guardian prompt"
    },
    {
      "path": "ADRs/084_semantic_entropy_tda_gating.md",
      "note": "Core Protocol 084"
    },
    {
      "path": "mcp_servers/learning/operations.py",
      "note": "Learning Operations Logic"
    }
  ]
}
```

---

## File: tools/standalone/context-bundler/base-manifests/base-learning-audit-core.json
**Path:** `tools/standalone/context-bundler/base-manifests/base-learning-audit-core.json`
**Note:** (Expanded from directory)

```json
{
    "title": "Learning Audit Core Context",
    "description": "Base manifest for Learning Audit packets (Protocol 128). Contains stable core context for Red Team review.",
    "files": [
        {
            "path": "README.md",
            "note": "Project overview"
        },
        {
            "path": "IDENTITY/founder_seed.json",
            "note": "Identity anchor"
        },
        {
            "path": ".agent/learning/cognitive_primer.md",
            "note": "Cognitive primer"
        },
        {
            "path": ".agent/learning/guardian_boot_contract.md",
            "note": "Guardian contract"
        },
        {
            "path": ".agent/learning/learning_audit/learning_audit_core_prompt.md",
            "note": "Core audit prompt"
        },
        {
            "path": ".agent/learning/learning_audit/learning_audit_prompts.md",
            "note": "Audit prompts"
        },
        {
            "path": ".agent/rules/cognitive_continuity_policy.md",
            "note": "Continuity policy"
        },
        {
            "path": "01_PROTOCOLS/128_Hardened_Learning_Loop.md",
            "note": "Protocol 128"
        },
        {
            "path": "ADRs/071_protocol_128_cognitive_continuity.md",
            "note": "ADR 071"
        },
        {
            "path": "docs/prompt-engineering/sanctuary-guardian-prompt.md",
            "note": "Guardian prompt"
        },
        {
            "path": "docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd",
            "note": "Protocol 128 diagram"
        }
    ]
}
```

---

## File: tools/standalone/context-bundler/base-manifests/base-generic-file-manifest.json
**Path:** `tools/standalone/context-bundler/base-manifests/base-generic-file-manifest.json`
**Note:** (Expanded from directory)

```json
{
  "title": "[TARGET] Generic Bundle",
  "description": "Generic context bundle for [TARGET]",
  "files": [
    {
      "path": "README.md",
      "note": "Project overview"
    }
  ]
}
```

---

## File: tools/standalone/context-bundler/base-manifests/base-bootstrap-file-manifest.json
**Path:** `tools/standalone/context-bundler/base-manifests/base-bootstrap-file-manifest.json`
**Note:** (Expanded from directory)

```json
{
    "title": "[TARGET] Bootstrap Context Bundle",
    "description": "Fresh repository onboarding context for new developers/agents.",
    "files": [
        {
            "path": "README.md",
            "note": "Project overview"
        },
        {
            "path": "Makefile",
            "note": "Build commands"
        },
        {
            "path": ".agent/learning/cognitive_primer.md",
            "note": "Cognitive primer"
        },
        {
            "path": "docs/operations/BOOTSTRAP.md",
            "note": "Bootstrap guide"
        },
        {
            "path": "ADRs/071_protocol_128_cognitive_continuity.md",
            "note": "ADR 071"
        },
        {
            "path": "ADRs/089_modular_manifest_pattern.md",
            "note": "ADR 089"
        }
    ]
}
```

---

## File: tools/standalone/context-bundler/base-manifests/base-context-bundler-file-manifest.json
**Path:** `tools/standalone/context-bundler/base-manifests/base-context-bundler-file-manifest.json`
**Note:** (Expanded from directory)

```json
{
  "title": "Context Bundler Tool",
  "description": "A standalone utility to concatenate multiple source files into a single context bundle for LLM analysis.",
  "files": [
    {
      "path": "tools/standalone/context-bundler/UNPACK_INSTRUCTIONS.md",
      "note": "CRITICAL: How to use this bundle"
    },
    {
      "path": "tools/standalone/context-bundler/README.md",
      "note": "Documentation and Usage Guide"
    },
    {
      "path": "tools/standalone/context-bundler/INSTALL.md",
      "note": "Installation and Unpacking Instructions"
    },
    {
      "path": "tools/standalone/context-bundler/TOOL_INVENTORY.md",
      "note": "Bundle-Specific Tool Inventory"
    },
    {
      "path": "docs/tools/standalone/context-bundler/bundler-internal-logic.mmd",
      "note": "Architecture Diagram (Internal Logic)"
    },
    {
      "path": "docs/tools/standalone/context-bundler/setup-lifecycle-workflow.mmd",
      "note": "User Workflow Diagram (Lifecycle)"
    },
    {
      "path": "docs/tools/standalone/context-bundler/agent-unpacking-process.mmd",
      "note": "Agent Unpacking Flow Diagram"
    },
    {
      "path": "tools/retrieve/bundler/bundle.py",
      "note": "Core Bundling Logic"
    },
    {
      "path": "tools/retrieve/bundler/manifest_manager.py",
      "note": "Source Code: Manifest Manager"
    },
    {
      "path": "tools/investigate/utils/path_resolver.py",
      "note": "Source Code: Path Resolver Utility"
    },
    {
      "path": "tools/standalone/context-bundler/file-manifest-schema.json",
      "note": "Input Schema Definition"
    },
    {
      "path": "docs/diagrams/workflows/curate-bundle.mmd",
      "note": "Workflow Diagram: Curate Bundle"
    },
    {
      "path": "docs/tools/standalone/context-bundler/architecture.md",
      "note": "Architecture & Recursive Workflow Logic"
    },
    {
      "path": "tools/standalone/context-bundler/base-manifests",
      "note": "Base Manifest Templates (Standard Configurations)"
    }
  ]
}
```
<a id='entry-18'></a>
### Directory: .agent/skills/context-bundling
**Note:** Skill: Context Bundling Instructions
> 📂 Expanding contents of `.agent/skills/context-bundling`...

---

## File: .agent/skills/context-bundling/SKILL.md
**Path:** `.agent/skills/context-bundling/SKILL.md`
**Note:** (Expanded from directory)

```markdown
---
name: context-bundling
description: Create technical bundles of code, design, and documentation for external review or context sharing. Use when you need to package multiple project files into a single Markdown file while preserving folder hierarchy and providing contextual notes for each file.
---

# Context Bundling Skill 📦

## Overview
This skill centralizes the knowledge and workflows for creating "Context Bundles" using the project's internal bundling tools. These bundles are essential for sharing large amounts of code and design context with other AI agents or for human review.

## Key Tools
- **Manifest Manager**: `tools/retrieve/bundler/manifest_manager.py` (Handles manifest creation and file management)
- **Bundler Engine**: `tools/retrieve/bundler/bundle.py` (Performs the actual Markdown generation)

## Core Workflow: Custom Temporary Bundles
When you need to create a one-off bundle for a specific task (like a Red Team review):

### 1. Initialize a Temporary Manifest
Always create temporary manifests in the `temp/` directory to keep the main tool configuration clean.
```bash
python3 tools/retrieve/bundler/manifest_manager.py --manifest temp/my_manifest.json init --type generic --bundle-title "Bundle Title"
```
> [!IMPORTANT]
> Global flags like `--manifest` MUST come **BEFORE** the subcommand (`init`, `add`, `bundle`).

### 2. Add Relevant Files
Add design docs, source code, and custom prompts to the manifest.
```bash
python3 tools/retrieve/bundler/manifest_manager.py --manifest temp/my_manifest.json add --path "docs/design.md" --note "Primary design"
```

### 3. Generate the Bundle
Compile the files into a single Markdown artifact.
```bash
python3 tools/retrieve/bundler/manifest_manager.py --manifest temp/my_manifest.json bundle --output temp/my_bundle.md
```

## Best Practices
1. **Contextual Notes**: Always provide a `--note` when adding files to help the recipient understand why that specific file is included.
2. **Cleanup**: Mention in your walkthrough that temporary files in `temp/` can be safely deleted after the bundle is used.
3. **Red Team Prompts**: When bundling for review, always include a specialized "Red Team Prompt" (e.g., `docs/architecture/red-team-*.md`) to guide the external LLM's review process.

## Manifest Schema (Reference)
If you need to manually edit a manifest:
```json
{
  "title": "Bundle Title",
  "description": "Context description",
  "files": [
    {
      "path": "path/to/file.ts",
      "note": "Description of why this file is here"
    }
  ]
}
```

```

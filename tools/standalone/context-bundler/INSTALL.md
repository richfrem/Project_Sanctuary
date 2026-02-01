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

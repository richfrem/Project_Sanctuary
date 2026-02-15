# 📦 Bundle Unpacking Protocol
> **🛑 STOP & READ**: Use this protocol to hydrate the tools in this bundle.

## Extraction Logic
1.  **Scan** this document for sections marked with **Path:** metadata (e.g., `**Path:** scripts/run.py`).
2.  **Extract** the code block content immediately following the path.
3.  **Save** the content to the specified filename (relative to your desired tool root).

## ⚠️ Critical Setup Step
After extracting the files, you **MUST** run the mapper script to initialize the inventory:
```bash
python map_repository_files.py
```
*Without this step, the auto-fixer will not work.*

## 📄 File List
The following files are included in this bundle:
- `README.md`
- `INSTALL.md`
- `check_broken_paths.py`
- `map_repository_files.py`
- `smart_fix_links.py`
- *Diagrams for context*

*(See standard Agent Unpacking Process diagram in the bundle for visuals)*

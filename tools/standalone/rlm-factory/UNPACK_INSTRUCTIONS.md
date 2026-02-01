# 📦 Bundle Unpacking Protocol
> **🛑 STOP & READ**: Use this protocol to hydrate the tools in this bundle.

## Extraction Logic
1.  **Scan** this document for sections marked with **Path:** metadata.
2.  **Extract** the code block content immediately following the path.
3.  **Save** the content to the specified filename (relative to your desired tool root).

## ⚠️ Critical Setup Step
After extracting the files, you **MUST** install dependencies:

```bash
pip install -r requirements.txt
```

And configure your environment:
1.  Create `.env`
2.  Set `OLLAMA_HOST` (if not default)
3.  Ensure Ollama is running (`ollama serve`).

## 📄 File List
The following files are included in this bundle:
- `distiller.py`
- `inventory.py`
- `query_cache.py`
- `cleanup_cache.py`
- `requirements.in` / `requirements.txt`
- `prompt.md`
- `research/summary.md`
- *Diagrams*

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
2.  Set `VECTOR_DB_PATH` and `VECTOR_DB_COLLECTION`
3.  (Optional) Ensure `RLM_CACHE_PATH` is managed via manifest (`tools/standalone/rlm-factory/manifest-index.json`).

## 📄 File List
The following files are included in this bundle:
- `ingest.py`
- `ingest_code_shim.py`
- `query.py`
- `cleanup.py`
- `requirements.in` / `requirements.txt`
- `prompt.md`
- *Diagrams*

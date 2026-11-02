# Hugging Face Dataset Integration Architecture

## Boundary & Delegation Invariant

`preserving-soul-traces` manages the domain logic of Sanctuary memory extraction, schema validation, and dataset preparation. It **NEVER** re-implements Hugging Face network APIs, OAuth, or exponential backoff.

All remote operations are delegated to the global APM skill `managing-huggingface-assets`.

```mermaid
graph LR
    A[Workspace Markdown] -->|trace_extractor.py| B[huggingface/data/soul_traces.jsonl]
    B -->|sync_soul_dataset.py| C[managing-huggingface-assets]
    C -->|hf_upload.py with backoff| D[Hugging Face Hub / Project_Sanctuary_Soul]
```

## Dataset Workspace Layout

```
huggingface/
├── README.md                # Dataset card with discovery tags and CC0 license
├── data/
│   └── soul_traces.jsonl    # Canonical JSONL training split
├── lineage/                 # Retrospective session summaries
└── metadata/
    ├── README.md            # Metadata documentation
    └── manifest.json        # Snapshot count, last_updated, and SHA256 digest
```

## Remote Repository Target

- **Repository**: `richfrem/Project_Sanctuary_Soul`
- **Type**: `dataset`
- **License**: CC0 1.0 (Public Domain Dedication)
- **Tags**: `cognitive-lineage`, `reasoning-traces`, `project-sanctuary`, `synthetic-memory`

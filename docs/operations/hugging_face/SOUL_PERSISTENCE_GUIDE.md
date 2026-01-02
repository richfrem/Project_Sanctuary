# Hugging Face Dataset: Source-to-Destination Mapping

## 1. Hugging Face Repository Structure

Based on the actual state of the [Project_Sanctuary_Soul](https://huggingface.co/datasets/richfrem/Project_Sanctuary_Soul) repository:

```text
richfrem/Project_Sanctuary_Soul/ (main)
├── data/           # soul_traces.jsonl, .gitkeep
├── lineage/        # seal_YYYYMMDD_HHMMSS_learning_package_snapshot.md, .gitkeep
├── metadata/       # manifest.json, .gitkeep
├── .gitattributes  # LFS configuration
└── README.md       # Primary Dataset Card
```

## 2. Repository Mapping

The Hugging Face repository uses the following structure and sources.

### A. Core Directories
| Local Source | Hub Destination | Role on Hugging Face |
| :--- | :--- | :--- |
| `.agent/learning/learning_package_snapshot.md` | **`lineage/`** | Individually timestamped reasoning trace snapshots. |
| (Runtime JSONL extraction) | **`data/`** | Machine-readable `soul_traces.jsonl` file. |
| `hugging_face_dataset_repo/metadata/` | **`metadata/`** | The staging area for provenance records and folder maintenance. |

### B. Specific File Mapping
| Local Source | Hub Destination | Role |
| :--- | :--- | :--- |
| **`hugging_face_dataset_repo/README.md`** | **`README.md` (Root)** | The primary **Dataset Card** landing page (includes Hub metadata). |
| **`hugging_face_dataset_repo/metadata/manifest.json`** | **`metadata/manifest.json`** | The provenance index tracking all snapshots. |
| **`hugging_face_dataset_repo/metadata/.gitkeep`** | **`metadata/.gitkeep`** | Maintenance file to preserve folder structure. |

---

## 3. Core Operations (Cortex Server)

The following methods in [operations.py](../../../mcp_servers/rag_cortex/operations.py) govern the persistence logic:

### `persist_soul(PersistSoulRequest)`
- **Role**: Tactical/Incremental update.
- **Internal Helper**: `_perform_soul_upload()` (Async engine for Hub communication).
- **Logic**: 
    1. Reads the latest Seal snapshot (`.md`).
    2. Calculates **Semantic Entropy (SE)** and **Alignment Scores** (ADR 084).
    3. Blocks persistence if SE > 0.95 (Hallucination) or SE < 0.2 (Rigidity).
    4. Hands off to `_perform_soul_upload()` for Hub broadcast.
    5. Appends record to `data/soul_traces.jsonl` and uploads the snapshot to `lineage/`.

### `persist_soul_full()`
- **Role**: Strategic/Full Re-Sync.
- **Logic**: 
    1. Scans the entire project (filtering for permitted files).
    2. Regenerates the comprehensive `soul_traces.jsonl` with updated ADR 084 scores for every record.
    3. Synchronizes the local `hugging_face_dataset_repo/data/` folder with the Hub's `data/` root via `upload_folder`.

### `perform_soul_upload()` (Internal Engine)
- **Role**: The core asynchronous bridge to the Hugging Face Hub API.
- **Functionality**:
    - **Idempotency**: Runs `ensure_dataset_structure()` and `ensure_dataset_card()`.
    - **Asset Handling**: Uploads files to `lineage/`, `data/`, or the root `README.md` depending on the request type.
    - **Protocol 128 Alignment**: Ensures the resulting Hub state follows the mandated ADR 081 structure.

---

## 4. CLI Mapping (The User Interface)

| CLI Command | Server Method | Result on Hub |
| :--- | :--- | :--- |
| `persist-soul` | `persist_soul` | Updates `lineage/` snapshot and appends to JSONL. |
| `persist-soul-full` | `persist_soul_full` | Rewrites the entire `data/soul_traces.jsonl` genome. |

---

## 5. Operational Routines

### **Updating the Landing Page**
To update the main landing page of the Hugging Face repo:
1.  Edit the project's root `README.md`.
2.  Run `scripts/hf_decorate_readme.py` to prepare the staged copy with metadata.
3.  Run `scripts/hf_upload_assets.py` to push the staged copy to the Hub root.

### **Recording a Session (The Seal)**
1.  Run `scripts/cortex_cli.py snapshot --type seal` to lock in the markdown summary.
2.  Run `scripts/cortex_cli.py persist-soul` to broadcast the summary and the data record to the Hub.

### **The Metadata Staging Folder**
The directory `hugging_face_dataset_repo/metadata/` is strictly for the Hub's **`metadata/`** folder. 
*   It does **not** update the project root.
*   It is **gitignored** and must be handled with care to avoid local state loss.
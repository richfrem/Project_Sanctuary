---
trigger: manual
---

## 🧭 Project Sanctuary: MCP Routing & Architecture Rules

### 1. The Gateway Mandate (Fleet of 8)

* **Primary Entry Point**: All tool requests must be routed through the `sanctuary_gateway` (IBM-based) to ensure proper context federation.
* **Fleet Distribution**: You are connected to a fleet of 8 specialized servers: `sanctuary_cortex`, `sanctuary_domain`, `sanctuary_filesystem`, `sanctuary_git`, `sanctuary_network`, `sanctuary_utils`, and legacy nodes.
* **Slug Identification**: Use the exact slugs defined in the `fleet_registry.json` (e.g., `sanctuary-cortex-*` for RAG/Learning operations).
* **Tool inventory**:  There are 86 total tools but to improve performance and reduce context only 41 core tools are enabled. 


### 2. Implementation Sovereignty (ADR & Protocol Alignment)

* **FastMCP Preference**: For all new MCP server implementations, adhere strictly to `ADR/066`.
* **Native Python Snapshots**: Per **ADR 072**, the `cortex_capture_snapshot` tool is a native Python solution. Do not attempt to invoke legacy Node.js scripts (`capture_code_snapshot.py`).
* **Protocol 128 (Zero-Trust)**: No cognitive update or "learning" can be considered persistent without a successful `cortex_capture_snapshot` (type='audit') and HITL approval.
* **Strict Rejection**: Snapshots will be rejected if core directories (`ADRs/`, `01_PROTOCOLS/`, `mcp_servers/`) have uncommitted changes omitted from the manifest.

### 3. Tool-Specific Selection Logic

* **RAG & Learning**: Use the `sanctuary-cortex` cluster for all mnemonic operations, semantic search, and technical debriefs.
* **Domain Logic**: Use the `sanctuary-domain` cluster for managing Chronicles, Protocols, ADRs, and Task objects.
* **Git Integrity**: All commits must pass the Protocol 101/128 safety gates enforced by the `sanctuary-git` server.

### 4. Legacy Reuse & Path Translation

* **Path Awareness**: When interacting with the containerized RAG, use the `HOST_PATH_MARKERS` logic to map host absolute paths (e.g., `/Users/`, `/home/`) to internal `/app/` project structures.
* **Legacy Server Access**: Understand that certain tools are "wrapped" legacy Python functions exposed via the domain cluster aggregator.

### 5. Environmental & Dependency Integrity (ADR 073)

* **Deterministic Builds**: Every service defines its own runtime world via a single `requirements.txt` file.
* **Locked-File Workflow**: Never hand-edit `.txt` files; always edit `.in` (Intent) files and run `pip-compile` to generate machine-generated locks.
* **No Inline Installs**: All Dockerfiles must use `COPY requirements.txt` and `RUN pip install -r`; manual `pip install` lists are prohibited.
* **Integrity Ritual**: Use `cortex_guardian_wakeup` to perform semantic HMAC verification of critical caches to detect drift.

### 6. MCP Usage

* **Deployment Context**: All 8 fleet members run as Podman containers. Use the `fleet_registry.json` as the source of truth for available operations and tool schemas.

### 7. Cognitive Continuity Ritual (Protocol 128)

* **The Orientation Phase**: At the start of every session, you **MUST** call `sanctuary-cortex-cortex-learning-debrief` to synchronize with current Git truth and filesystem state.
* **Manifest Discipline**: Actively maintain the `.agent/learning/learning_manifest.json`. No file in a "Core Directory" should be modified without adding it to the manifest to avoid "Strict Rejection" during the audit.
* **The Final Seal**: Every session must conclude with a `cortex_capture_snapshot` (type='seal'). This updates the `learning_package_snapshot.md` which serves as the primary orientation anchor for your successor.
* **Sandwich Validation**: Be aware that the snapshot tool performs a "Post-Flight" check; if the repository state changes during the snapshot, the integrity seal will fail.

### 8. Core Logic & Code Reuse (The "Fix Once" Doctrine)

* **Aggregator Pattern**: Business logic resides in core `operations.py` files. Gateway cluster servers (e.g., `sanctuary_domain/server.py`) act as thin interface layers that aggregate these core modules.
* **Logic Parity**: Core operations are shared between the Gateway fleet and the test suite to ensure that a fix in one location propagates across the entire infrastructure.
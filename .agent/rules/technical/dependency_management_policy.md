---
trigger: manual
---

## 🐍 Project Sanctuary: Python Dependency & Environment Rules

### 1. Core Mandate: One Runtime World

* 
**Service Sovereignty**: Every service (e.g., `sanctuary_cortex`, `sanctuary_git`) owns its own runtime environment expressed through a single `requirements.txt` file.

* **Parity Requirement**: The execution environment (Docker, Podman, `.venv`) must not change the dependency logic. You must install from the same locked artifact regardless of where the code runs.

* 
**Prohibition of Manual Installs**: You are strictly forbidden from running `pip install <package>` directly in a terminal or adding it as a manual `RUN` command in a Dockerfile.


### 2. The Locked-File Ritual (Intent vs. Truth)

* **Human Intent (`.in`)**: All dependency changes must start in the `.in` file (e.g., `requirements.in`). This is where you declare high-level requirements like `fastapi` or `langchain`.

* **Machine Truth (`.txt`)**: The `.txt` file is a machine-generated lockfile created by `pip-compile`. It contains the exact versions and hashes of every package in the dependency tree.

* **The Compilation Step**: After editing a `.in` file, you **must** run the compilation command to synchronize the lockfile:

`pip-compile <service>/requirements.in --output-file <service>/requirements.txt`.


### 3. Tiered Dependency Hierarchy

* 
**Tier 1: Common Core**: Shared baseline dependencies (e.g., `mcp`, `fastapi`, `pydantic`) are managed in `mcp_servers/gateway/requirements-core.in`.

* 
**Tier 2: Specialized extras**: Service-specific heavy lifters (e.g., `chromadb` for Cortex) are managed in the individual service's `.in` file.

* 
**Tier 3: Development Tools**: Tools like `pytest`, `black`, or `ruff` belong exclusively in `requirements-dev.in` and must never be installed in production containers.


### 4. Container & Dockerfile Constraints

* **Declarative Builds**: Dockerfiles must only use `COPY requirements.txt` followed by `RUN pip install -r`. This ensures the container is a perfect mirror of the verified local lockfile.

* 
**Cache Integrity**: Do not break Docker layer caching by copying source code before installing requirements.


### 5. Dependency Update Workflow

1. 
**Declare**: Add the package name to the relevant `.in` file.

2. 
**Lock**: Run `pip-compile` to generate the updated `.txt` file.

3. 
**Sync**: Run `pip install -r <file>.txt` in your local environment.

4. 
**Verify**: Rebuild the affected Podman container to confirm the build remains stable.

5. 
**Commit**: Always commit **both** the `.in` and `.txt` files to Git together.
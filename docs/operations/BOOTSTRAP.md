# 🏗️ BOOTSTRAP: Initial Project Setup (Cross-Platform)

This guide walks you through the initial setup of a fresh Project Sanctuary clone. It is designed to work on **macOS, Linux, and Windows (via WSL2)**, following the **ADR 073** standard for tiered dependency management.

---

## ⚠️ PREREQUISITE: Sanctuary Gateway

> [!CAUTION]
> **This project depends on the [Sanctuary Gateway](https://github.com/richfrem/sanctuary-gateway) being installed and running FIRST.** The fleet cannot register without it.

Before proceeding with this guide, you must complete the gateway setup:

1. **Clone the Gateway Repo**:
   ```bash
   git clone https://github.com/richfrem/sanctuary-gateway.git
   cd sanctuary-gateway
   ```

2. **Run the Gateway Setup Script**:
   This script builds the container, bootstraps the admin user, and generates your API token.
   ```bash
   python3 setup/recreate_gateway.py
   ```

3. **Copy the Generated Token**:
   The script outputs `MCPGATEWAY_BEARER_TOKEN` and saves it to the gateway's `.env` file.
   - **macOS/Linux**: Add this token to your `~/.zshrc` or `~/.bashrc`.
   - **Windows/WSL2**: Add the token to your **Windows User Environment Variables** and ensure `WSLENV` includes `MCPGATEWAY_BEARER_TOKEN/u`.

4. **Verify the Gateway is Running**:
   ```bash
   curl -ks https://localhost:4444/health
   ```

---

## 🟢 Phase 0: Environment Verification

Project Sanctuary requires a Unix-like environment for its MCP servers and ML dependencies.

1. **OS**: macOS (13+), Linux (Ubuntu 22.04+), or Windows (WSL2 with Ubuntu 22.04+).
2. **Python**: `python3 --version` should be 3.11 or higher.
3. **Container Engine**: Podman (v4+) should be installed and running (macOS: `brew install podman && podman machine init && podman machine start`; WSL2: follow Podman docs).
4. **Ollama**: Install and start Ollama for local LLM inference:
   - **macOS**: `brew install ollama && ollama serve`
   - **Linux/WSL2**: Follow [ollama.ai](https://ollama.ai) installation guide
   - **Verify**: `curl http://localhost:11434/api/tags` should return JSON

---

## 🔵 Phase 1: Virtual Environment (The Sanctuary)

1. **Clone the Repo**:
   ```bash
   git clone <repo-url>
   cd Project_Sanctuary
   ```

2. **Run the Bootstrap Sequence**:
   The `Makefile` creates the `.venv` and installs the locked dependency tiers.
   ```bash
   make bootstrap
   ```

3. **Activate the Environment**:
   ```bash
   source .venv/bin/activate
   ```

---

## 🟡 Phase 2: Dependency Tiers (ADR 073)

The `make bootstrap` command automatically installs the first two tiers. You can use specific targets for maintenance:

### Tier 1 & 2: Runtime (Core + Services)
```bash
make install-env
```

### Tier 3: Development Tools
Installs testing, linting, and formatting tools (pytest, ruff, black).
```bash
make install-dev
```

---

## 🔴 Phase 3: Secrets & Gateway Configuration

1. **Configure API Keys**:
   Set your API keys as environment variables in your shell profile (`.zshrc`, `.bashrc`) or use `.env` (not committed).
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`
   - `HUGGING_FACE_HUB_TOKEN`

   > [!TIP]
   > **Windows/WSL2 Users**: Set variables in Windows and use `WSLENV` to pass them through automatically.

---

## 🚀 Phase 4: Podman Fleet Deployment

Once your local dependencies are installed and secrets are configured, you can deploy the "Fleet of 8" MCP infrastructure.

1. **Ensure the Gateway is Running**:
   The Sanctuary Gateway (Port 4444) should be running as a separate service (managed in the `sanctuary-gateway` repo).

2. **Deploy the Fleet**:
   Use the unified Makefile to pull images, build containers, and register them with the gateway.
   ```bash
   make up
   ```

3. **Verify Fleet Health**:
   Check if all 8 containers are running and healthy:
   ```bash
   make status
   ```

4. **Run Connectivity Tests**:
   Ensure the gateway can communicate with the newly deployed servers:
   ```bash
   make verify
   ```

---

## 🧠 Phase 5: Knowledge Base Initialization (ChromaDB)

After the fleet is running, initialize the vector database with project content:

1. **Verify ChromaDB Container**:
   The `sanctuary_vector_db` container should be running (check with `make status`).

2. **Run Initial Ingestion**:
   Ingest the project's knowledge base into ChromaDB:
   ```bash
   python3 scripts/cortex_cli.py ingest --full
   ```

3. **Verify Ingestion**:
   ```bash
   python3 scripts/cortex_cli.py query "What is Project Sanctuary?"
   ```

> [!TIP]
> For incremental updates after editing documentation, use:
> ```bash
> python3 scripts/cortex_cli.py ingest --incremental --hours 24
> ```

---

## 🛡️ Troubleshooting & Maintenance

- **Detailed Operations**: For granular control, targeted rebuilds, and deep-dive maintenance, refer to the [Podman Operations Guide](docs/operations/processes/PODMAN_OPERATIONS_GUIDE.md).
- **Missing .txt files**: If a collaborator added a dependency to a `.in` file but didn't commit the `.txt`, run `make compile`.
- **WSLENV Errors**: Ensure your Windows environment variables are named exactly as expected by `mcp_servers/lib/env_helper.py`.
- **Podman Context**: If the Gateway cannot connect to containers, verify you are not mixing Docker and Podman contexts.
- **Ollama Not Responding**: Ensure `ollama serve` is running in a separate terminal or as a background service.
- **ChromaDB Empty**: If queries return no results, re-run `python3 scripts/cortex_cli.py ingest --full`.

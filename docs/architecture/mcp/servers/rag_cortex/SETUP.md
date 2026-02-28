# RAG Cortex Setup Guide

**Last Updated:** 2025-12-20  
**Status:** Canonical

---

> [!IMPORTANT]
> **🚀 Unified Fleet Deployment (ADR 065)**  
> As of December 2025, Project Sanctuary uses a **unified Makefile-based deployment** for all Agent Plugin Integration servers.  
> Instead of manual `podman` commands, use:
> ```bash
> make up      # Deploy entire Fleet of 8 (includes RAG Cortex)
> make down    # Stop all services
> make status  # Check fleet health
> ```
> See [[PODMAN_OPERATIONS_GUIDE|PODMAN_STARTUP_GUIDE.md]] for complete workflow.
>
> The manual `podman` commands below are preserved for reference and troubleshooting only.

---

## Overview

This guide covers the complete setup and operation of the RAG Cortex Agent Plugin Integration server with ChromaDB running as a containerized service via Podman (per [[034_containerize_mcp_servers_with_podman|ADR 034]]).

The RAG Cortex provides retrieval-augmented generation capabilities for Project Sanctuary, managing the knowledge base, vector embeddings, and semantic search.

---

## Prerequisites

### General Agent Plugin Integration Prerequisites

Before setting up RAG Cortex, ensure you have completed the general Agent Plugin Integration prerequisites:

**📖 See: [[prerequisites|Agent Plugin Integration Server Prerequisites]]**

Key requirements:
- **Podman** installed and running (Podman Desktop recommended)
- **Python 3.11+** with virtual environment activated
- **Project dependencies** installed from `requirements.txt`

### System Requirements

- **CPU:** 4+ cores recommended
- **RAM:** 8GB minimum, 16GB recommended for large ingestion
- **Disk:** 10GB+ free space for vector database and container images
- **Network:** Port 8000 available for ChromaDB service

## Automated Setup Validation

Run the comprehensive validation script to verify your setup:

```bash
source .venv/bin/activate
python tests/mcp_servers/rag_cortex/test_setup_validation.py
```

This script will:
1. ✅ Verify ChromaDB container is running
2. ✅ Test database connectivity  
3. ✅ Check if data exists
4. ✅ Perform full ingestion if needed (~5 minutes for 431 documents)
5. ✅ Run sample queries to validate content

**Expected output:**
```
🚀 RAG CORTEX SETUP VALIDATION

✓ ChromaDB container already running and healthy
✓ Connected successfully!
✓ Contains 5636 chunks
✓ Query successful!

✅ ALL TESTS PASSED - RAG Cortex is ready!
```

## Manual Verification Steps

If you prefer to verify manually:

### Step 1: Verify Podman Installation

Verify Podman is properly installed and running:

```bash
# Check Podman version
podman --version
# Expected: podman version 5.7.0 (or later)

# Check machine status
podman machine list
# Should show: Currently running

# Test with hello-world (IMPORTANT: Do this first!)
podman run --rm hello-world
# Should download image and show "Hello from Podman!" message
```

**If Podman is not installed**, follow the installation guide in [[prerequisites#1-podman-containerization|prerequisites.md]]:

```bash
# macOS: Download Podman Desktop
# https://podman-desktop.io/downloads

# Or via Homebrew
brew install podman

# Initialize and start
podman machine init
podman machine start
```

### Step 2: Python Environment Verification

```bash
# Verify Python version
python3 --version
# Expected: Python 3.11.x or later

# Verify virtual environment is activated
which python
# Should show: /path/to/Project_Sanctuary/.venv/bin/python

# Verify dependencies installed
pip show langchain-chroma langchain-nomic
```

### Step 3: Port Configuration

ChromaDB uses port 8000 by default. If this port is already in use on your system, you can configure a different port:

**Check if port 8000 is available:**
```bash
lsof -i :8000
# If this returns nothing, port 8000 is available
# If it shows a process, you'll need to use a different port
```

**To use a different port:**

1. Edit `.env` file:
   ```bash
   CHROMA_HOST=localhost  # Use localhost when connecting from host machine
   CHROMA_PORT=8000       # Container internal port (keep as 8000)
   PODMAN_HOST_PORT=9000  # Change this to your desired host port
   ```

2. Update `docker-compose.yml`:
   ```yaml
   ports:
     - "9000:8000"  # host_port:container_port
   ```

3. Or use Podman directly with custom port:
   ```bash
   podman run -d --name sanctuary_vector_db \
     -p 9000:8000 \  # Use your custom port here
     -v ./.vector_data:/chroma/chroma:Z \
     chromadb/chroma:latest
   ```

---

## Environment

### 3. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

> [!IMPORTANT]
> **Critical Configuration**: Set `CHROMA_HOST=localhost` in your `.env` file.
> The `.env.example` file may show `vector_db` (for docker-compose networking),
> but for local development you **must** use `localhost`.

**Required settings:**
```bash
CHROMA_HOST=localhost  # MUST be localhost for local development
CHROMA_PORT=8000
PODMAN_HOST_PORT=8000  # Change if port 8000 is in use
CHROMA_DATA_PATH=.vector_data # Local path for persistent vector storage

# Collection names
CHROMA_CHILD_COLLECTION=child_chunks_v5
CHROMA_PARENT_STORE=parent_documents_v5
```

> [!TIP]
> - If port 8000 is already in use, change `PODMAN_HOST_PORT` to an available port (e.g., 9000) and update the port mapping in `docker-compose.yml` accordingly.
> - If you want to store vector data in a different location, change `CHROMA_DATA_PATH` to your desired path (absolute or relative to project root).

## Initial Setup

### Step 1: Create Data Directory

```bash
mkdir -p .vector_data
```

This directory will be bind-mounted into the ChromaDB container for data persistence.

### Step 2: Start Agent Plugin Integration Services

Using Podman Compose (Docker Compose compatible):

```bash
# Start both critical Agent Plugin Integration services (unified application stack)
podman-compose up -d vector_db ollama_model_mcp
```

> [!TIP]
> **Unified Launch**: This command starts both the RAG Cortex (vector_db) and Forge LLM (ollama_model_mcp) services together, ensuring the complete Agent Plugin Integration infrastructure is available.

Or using Podman directly for vector_db only:

```bash
podman run -d \
  --name sanctuary_vector_db \
  -p 8000:8000 \
  -v ./.vector_data:/chroma/chroma:Z \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  --restart unless-stopped \
  chromadb/chroma:latest
```

> [!NOTE]
> The `:Z` flag on the volume mount is important for SELinux systems. On macOS, it's optional but harmless.

### Step 3: Verify Service Health

Check that ChromaDB is running:

```bash
# Check container status
podman ps

# Check health endpoint
#curl http://localhost:8000/api/v1/heartbeat
curl http://localhost:8000/api/v2/heartbeat
```

Expected response: `{"nanosecond heartbeat": <timestamp>}`

### Step 4: Populate the Database

Run the full ingestion script to populate ChromaDB with the Cognitive Genome:

```bash
python3 scripts/cortex_ingest_full.py
```

This will:
- Load all markdown files from project directories
- Create embeddings using Nomic
- Store chunks in ChromaDB via network connection
- Store parent documents in `.vector_data/parent_documents_v5/`

> [!TIP]
> Initial ingestion can take 10-30 minutes depending on project size and hardware.

### Step 5: Verify Database Population

Check database statistics:

```bash
python3 scripts/cortex_stats.py
```

Or via Agent Plugin Integration (if orchestrator is running):
```python
# Via Agent Plugin Integration client
cortex_get_stats(include_samples=True)
```

## Daily Operations

### Starting the Service

```bash
# Start both Agent Plugin Integration services (recommended)
podman-compose up -d vector_db ollama_model_mcp
# or start individual services
podman start sanctuary_vector_db
podman start sanctuary_ollama
```

### Stopping the Service

```bash
podman-compose down
# or
podman stop sanctuary_vector_db
```

### Viewing Logs

```bash
podman logs -f sanctuary_vector_db
```

### Incremental Updates

After adding new documents to the project:

```bash
python scripts/cortex_ingest_incremental.py path/to/new/file.md
```

Or via Agent Plugin Integration:
```python
cortex_ingest_incremental(file_paths=["path/to/new/file.md"])
```

## Troubleshooting

### Service Won't Start

**Check Podman machine status:**
```bash
podman machine list
podman machine start
```

**Check port availability:**
```bash
lsof -i :8000
```

**Check container logs:**
```bash
podman logs sanctuary_vector_db
```

### Connection Refused

**Verify environment variables:**
```bash
grep CHROMA .env
```

**Verify service is listening:**
```bash
curl http://localhost:8000/api/v2/heartbeat
```

**Check network configuration:**
```bash
podman inspect sanctuary_vector_db | grep IPAddress
```

### Data Persistence Issues

**Verify bind mount:**
```bash
podman inspect sanctuary_vector_db | grep -A 5 Mounts
```

**Check directory permissions:**
```bash
ls -la .vector_data/
```

**Verify data exists:**
```bash
ls -la .vector_data/child_chunks_v5/
ls -la .vector_data/parent_documents_v5/
```

## Advanced Configuration

### Custom Collection Names

Edit `.env` to use different collection names:
```bash
CHROMA_CHILD_COLLECTION=my_custom_chunks
CHROMA_PARENT_STORE=my_custom_parents
```

### Resource Limits

Add resource constraints to `docker-compose.yml`:
```yaml
services:
  vector_db:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Network Configuration

To run ChromaDB on a different port:

1. Update `docker-compose.yml`:
   ```yaml
   ports:
     - "9000:8000"  # host:container
   ```

2. Update `.env`:
   ```bash
   CHROMA_PORT=9000
   ```

## Migration from Legacy Setup

If you have existing ChromaDB data at legacy paths (e.g., `mnemonic_cortex/chroma_db` or `data/cortex/chroma_db`):

### Option A: Copy Existing Data

```bash
mkdir -p .vector_data
# Adjust source path to match your legacy location
cp -r data/cortex/chroma_db/* .vector_data/
```

### Option B: Re-ingest (Recommended)

```bash
# Start fresh with network architecture
podman-compose up -d vector_db
python scripts/cortex_ingest_full.py
```

## References

- [[034_containerize_mcp_servers_with_podman|ADR 034: Containerize Agent Plugin Integration Servers with Podman]]
- [Podman Documentation](https://docs.podman.io/)
- [Podman Desktop](https://podman-desktop.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

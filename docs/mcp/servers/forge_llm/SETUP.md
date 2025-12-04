# Ollama Model Service Setup Guide

**Last Updated:** 2025-12-04  
**Status:** Canonical  
**Related Task:** T093

---

## Overview

This guide covers the setup and operation of the Ollama Model Service running as a containerized service via Podman/Docker Compose. This service provides network-accessible LLM inference for the Forge LLM MCP, following the same architecture pattern as the RAG Cortex MCP's ChromaDB service.

---

## Prerequisites

### General MCP Prerequisites

Before setting up the Ollama service, ensure you have completed the general MCP prerequisites:

**📖 See: [MCP Server Prerequisites](../../prerequisites.md)**

Key requirements:
- **Podman** or **Docker** installed and running
- **GPU drivers** installed (NVIDIA CUDA for GPU acceleration)
- **Python 3.11+** with virtual environment activated

### System Requirements

- **CPU:** 4+ cores recommended
- **RAM:** 16GB minimum, 32GB recommended for 7B models
- **GPU:** NVIDIA GPU with 8GB+ VRAM recommended (CPU inference supported but slow)
- **Disk:** 10GB+ free space for model storage
- **Network:** Port 11434 available for Ollama service

---

## Environment Configuration

### Configure Environment Variables

Ensure your `.env` file contains the Ollama configuration:

```bash
# Model Context Protocol (Model MCP) Configuration - Network Connection
# Ollama runs as a Podman container service (see docker-compose.yml)
# Use 'localhost' for local development, 'ollama-model-mcp' for docker-compose networking
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=Sanctuary-Qwen2-7B:latest
```

> [!IMPORTANT]
> **Critical Configuration**: Set `OLLAMA_HOST=http://localhost:11434` in your `.env` file for local development.
> For inter-container communication (e.g., from other services in docker-compose), use `http://ollama-model-mcp:11434`.

---

## Initial Setup

### Step 1: Start Ollama Service

Using Docker Compose (recommended):

```bash
docker-compose up -d ollama-model-mcp
```

Or using Podman Compose:

```bash
podman-compose up -d ollama-model-mcp
```

### Step 2: Verify Service Health

Check that Ollama is running:

```bash
# Check container status
docker ps | grep sanctuary-ollama-mcp
# or
podman ps | grep sanctuary-ollama-mcp

# Check Ollama API version
curl http://localhost:11434/api/version
```

Expected response: `{"version":"0.x.x"}`

### Step 3: Verify Model Availability

The container automatically pulls the configured model on first start. Check model availability:

```bash
# List available models
curl http://localhost:11434/api/tags

# Should include Sanctuary-Qwen2-7B:latest
```

> [!NOTE]
> Initial model download can take 5-15 minutes depending on your internet connection (model is ~4-5GB).

---

## Manual Podman Setup (Alternative)

If not using docker-compose, run Ollama manually with Podman:

```bash
# IMPORTANT: Ensure OLLAMA_MODEL is set in the shell before running
podman run -d \
  --name sanctuary-ollama-mcp \
  -p 11434:11434 \
  -v ./ollama_models:/root/.ollama:Z \
  -e OLLAMA_HOST=0.0.0.0 \
  -e OLLAMA_MODEL=${OLLAMA_MODEL:-Sanctuary-Qwen2-7B:latest} \
  --device=all \
  --restart=unless-stopped \
  ollama/ollama:latest \
  /bin/sh -c "ollama serve & sleep 10 && ollama pull ${OLLAMA_MODEL:-Sanctuary-Qwen2-7B:latest} && wait -n"
```

> [!TIP]
> The `:Z` flag on the volume mount is important for SELinux systems. On macOS, it's optional but harmless.

---

## Daily Operations

### Starting the Service

```bash
docker-compose up -d ollama-model-mcp
# or
docker start sanctuary-ollama-mcp
# or
podman start sanctuary-ollama-mcp
```

### Stopping the Service

```bash
docker-compose down
# or
docker stop sanctuary-ollama-mcp
# or
podman stop sanctuary-ollama-mcp
```

### Viewing Logs

```bash
docker logs -f sanctuary-ollama-mcp
# or
podman logs -f sanctuary-ollama-mcp
```

### Testing Inference

Test the model with a simple query:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "Sanctuary-Qwen2-7B:latest",
  "prompt": "What is Protocol 101?",
  "stream": false
}'
```

---

## Troubleshooting

### Service Won't Start

**Check Podman/Docker machine status:**
```bash
podman machine list
podman machine start
# or
docker info
```

**Check port availability:**
```bash
lsof -i :11434
```

**Check container logs:**
```bash
docker logs sanctuary-ollama-mcp
# or
podman logs sanctuary-ollama-mcp
```

### GPU Not Detected

**Verify NVIDIA drivers:**
```bash
nvidia-smi
```

**Check GPU access in container:**
```bash
docker exec sanctuary-ollama-mcp nvidia-smi
# or
podman exec sanctuary-ollama-mcp nvidia-smi
```

**For AMD GPUs**, modify the `docker-compose.yml` deploy section:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: amd
          count: all
          capabilities: [gpu]
```

### Model Download Fails

**Check internet connectivity:**
```bash
curl https://ollama.ai
```

**Manually pull model:**
```bash
docker exec sanctuary-ollama-mcp ollama pull Sanctuary-Qwen2-7B:latest
# or
podman exec sanctuary-ollama-mcp ollama pull Sanctuary-Qwen2-7B:latest
```

### Connection Refused

**Verify environment variables:**
```bash
grep OLLAMA .env
```

**Verify service is listening:**
```bash
curl http://localhost:11434/api/version
```

**Check network configuration:**
```bash
docker inspect sanctuary-ollama-mcp | grep IPAddress
# or
podman inspect sanctuary-ollama-mcp | grep IPAddress
```

### Data Persistence Issues

**Verify bind mount:**
```bash
docker inspect sanctuary-ollama-mcp | grep -A 5 Mounts
# or
podman inspect sanctuary-ollama-mcp | grep -A 5 Mounts
```

**Check directory permissions:**
```bash
ls -la ollama_models/
```

**Verify model data exists:**
```bash
ls -la ollama_models/models/
```

---

## Advanced Configuration

### Using Different Models

To use a different model, update `.env`:

```bash
OLLAMA_MODEL=llama2:13b
# or
OLLAMA_MODEL=mistral:latest
```

Then restart the service:

```bash
docker-compose restart ollama-model-mcp
```

### Resource Limits

Add resource constraints to `docker-compose.yml`:

```yaml
services:
  ollama-model-mcp:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
        reservations:
          cpus: '2.0'
          memory: 8G
          devices:
            - driver: nvidia
              count: 1  # Use only 1 GPU
              capabilities: [gpu]
```

### Network Configuration

To run Ollama on a different port:

1. Update `docker-compose.yml`:
   ```yaml
   ports:
     - "12434:11434"  # host:container
   ```

2. Update `.env`:
   ```bash
   OLLAMA_HOST=http://localhost:12434
   ```

---

## Integration with Forge LLM MCP

Once the Ollama service is running, the Forge LLM MCP can connect to it using the configured `OLLAMA_HOST`.

**Verify integration:**

```python
# Via Forge LLM MCP
from mcp_servers.forge_llm import query_sanctuary_model

result = query_sanctuary_model(
    prompt="What is the Sanctuary's mission?",
    temperature=0.7
)
print(result)
```

---

## References

- [Task T093: Containerize Ollama Model Service](../../../TASKS/in-progress/093_containerize_ollama_model_service_podman.md)
- [Ollama Documentation](https://ollama.ai/docs)
- [Podman Documentation](https://docs.podman.io/)
- [Docker Documentation](https://docs.docker.com/)

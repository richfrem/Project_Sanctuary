# sanctuary_cortex Dockerfile
# Fleet of 8 - Container #5: RAG/Cortex engine
# Designed to run with context at project root

FROM python:3.11

WORKDIR /app

# Install build deps and curl (cmake removed as gpt4all compilation is no longer needed)
RUN apt-get update && apt-get install -y curl build-essential git git-lfs && rm -rf /var/lib/apt/lists/*

# Copy shared libraries (CRITICAL REQUIREMENT)
COPY mcp_servers/lib /app/mcp_servers/lib

# Copy the fleet cluster code
# Source (Host): mcp_servers/gateway/clusters/sanctuary_cortex (kebab-case)
# Dest (Container): /app/mcp_servers/gateway/clusters/sanctuary_cortex (snake_case)
COPY mcp_servers/gateway/clusters/sanctuary_cortex /app/mcp_servers/gateway/clusters/sanctuary_cortex

# Install Python dependencies
COPY mcp_servers/gateway/clusters/sanctuary_cortex/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
    langchain-huggingface \
    langchain-chroma \
    sentence-transformers \
    einops \
    chromadb \
    langchain-community \
    langchain \
    pydantic \
    fastapi \
    uvicorn \
    sse-starlette \
    httpx \
    ollama \
    openai

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "mcp_servers.gateway.clusters.sanctuary_cortex.server"]

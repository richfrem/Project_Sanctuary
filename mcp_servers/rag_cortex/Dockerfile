# sanctuary-cortex Dockerfile
# Fleet of 8 - Container #5: RAG/Cortex engine
# Designed to run with context at project root

FROM python:3.11-slim

WORKDIR /app

# Install build deps and curl
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Copy the shared lib and rag_cortex code
COPY mcp_servers/lib /app/mcp_servers/lib
COPY mcp_servers/rag_cortex /app/mcp_servers/rag_cortex

# Install Python dependencies (from requirements + extras)
COPY mcp_servers/rag_cortex/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sse-starlette \
    httpx \
    pydantic \
    chromadb

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "mcp_servers.rag_cortex.server"]

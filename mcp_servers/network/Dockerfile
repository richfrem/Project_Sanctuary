# sanctuary-network Dockerfile
# Fleet of 8 - Container #3: Network tools
# Designed to run with context at project root

FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the shared lib and network code
COPY mcp_servers/lib /app/mcp_servers/lib
COPY mcp_servers/network /app/mcp_servers/network

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sse-starlette \
    httpx \
    pydantic

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "mcp_servers.network.server"]

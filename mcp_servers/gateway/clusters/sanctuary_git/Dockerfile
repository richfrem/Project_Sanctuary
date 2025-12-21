# sanctuary_git Dockerfile
# Fleet of 8 - Container #4: Git workflow tools
# Designed to run with context at project root

FROM python:3.11-slim

WORKDIR /app

# Install git and curl
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Copy shared libraries (CRITICAL REQUIREMENT)
COPY mcp_servers/lib /app/mcp_servers/lib

# Copy the fleet cluster code
# Source (Host): mcp_servers/gateway/clusters/sanctuary_git (kebab-case)
# Dest (Container): /app/mcp_servers/gateway/clusters/sanctuary_git (snake_case for Python import)
COPY mcp_servers/gateway/clusters/sanctuary_git /app/mcp_servers/gateway/clusters/sanctuary_git

# Install Python dependencies (FastMCP for proper MCP protocol - ADR-066)
RUN pip install --no-cache-dir \
    fastmcp \
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

# Launch using the new fleet namespace path
CMD ["python", "-m", "mcp_servers.gateway.clusters.sanctuary_git.server"]

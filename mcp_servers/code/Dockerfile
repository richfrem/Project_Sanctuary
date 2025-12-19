FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (git is needed for some code ops if they use git commands, though Git MCP is separate)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up build context structure
COPY mcp_servers/lib /app/mcp_servers/lib
COPY mcp_servers/code /app/mcp_servers/code

# Install dependencies
# We use a minimal requirements list for code/filesystem ops
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sse-starlette \
    httpx \
    pydantic

# Check if specific requirements exist in the copied source (optional robust pattern)
# For now, we manually install the core libs needed for the server.py we refactored

ENV PYTHONPATH=/app
ENV PORT=8000

# Run module
CMD ["python", "-m", "mcp_servers.code.server"]

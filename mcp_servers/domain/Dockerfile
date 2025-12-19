FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# git: for version control operations if needed
# curl: for healthchecks
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Common Python Dependencies for all included MCPs
RUN pip install --no-cache-dir \
    fastmcp \
    pydantic \
    requests \
    python-dateutil \
    httpx

# Install Development Tools (hosted here for the 'code' MCP delegation)
RUN pip install --no-cache-dir \
    ruff \
    black \
    pylint \
    flake8 \
    mypy

# Copy all MCP servers to allow cross-module imports
# The build context must be the project root
COPY mcp_servers /app/mcp_servers

# Set Environment Variables
ENV PYTHONPATH=/app
ENV PROJECT_ROOT=/app
ENV DOMAIN_SERVER_PORT=8105

# Expose the port
EXPOSE 8105

# Run the Unified Server
CMD ["python", "mcp_servers/domain/unified_server.py"]

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY mcp_servers/gateway/clusters/sanctuary_filesystem/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set up build context structure
COPY mcp_servers/lib /app/mcp_servers/lib

# Copy the fleet cluster code
# Source (Host): mcp_servers/gateway/clusters/sanctuary_filesystem (kebab-case)
# Dest (Container): /app/mcp_servers/gateway/clusters/sanctuary_filesystem (snake_case)
COPY mcp_servers/gateway/clusters/sanctuary_filesystem /app/mcp_servers/gateway/clusters/sanctuary_filesystem


ENV PYTHONPATH=/app
ENV PORT=8000

# Run module
CMD ["python", "-m", "mcp_servers.gateway.clusters.sanctuary_filesystem.server"]

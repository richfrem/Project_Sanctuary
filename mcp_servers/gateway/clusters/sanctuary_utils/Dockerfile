# sanctuary_utils Dockerfile
# Fleet of 8 - Container #1: Low-risk utility tools
# Designed to run with context at project root

FROM python:3.11-slim

LABEL maintainer="Project Sanctuary"
LABEL description="Fleet of 8 - sanctuary_utils: Low-risk utility tools"
LABEL version="1.0.0"

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl git git-lfs && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY mcp_servers/gateway/clusters/sanctuary_utils/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy shared libraries (CRITICAL REQUIREMENT)
COPY mcp_servers/lib /app/mcp_servers/lib

# Copy the fleet cluster code
COPY mcp_servers/gateway/clusters/sanctuary_utils /app/mcp_servers/gateway/clusters/sanctuary_utils


ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server module
CMD ["python", "-m", "mcp_servers.gateway.clusters.sanctuary_utils.server"]

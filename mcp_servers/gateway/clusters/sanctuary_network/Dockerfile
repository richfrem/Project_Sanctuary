# sanctuary_network Dockerfile
# Fleet of 8 - Container #3: Network tools
# Designed to run with context at project root

FROM python:3.11

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl git git-lfs && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY mcp_servers/gateway/clusters/sanctuary_network/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the shared lib
COPY mcp_servers/lib /app/mcp_servers/lib

# Copy the fleet cluster code
# Source (Host): mcp_servers/gateway/clusters/sanctuary_network (kebab-case)
# Dest (Container): /app/mcp_servers/gateway/clusters/sanctuary_network (snake_case)
COPY mcp_servers/gateway/clusters/sanctuary_network /app/mcp_servers/gateway/clusters/sanctuary_network


ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "mcp_servers.gateway.clusters.sanctuary_network.server"]

FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Common Python Dependencies
# Install dependencies
COPY mcp_servers/gateway/clusters/sanctuary_domain/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy shared lib
COPY mcp_servers/lib /app/mcp_servers/lib

# Copy Bundled Logical Servers (The "Brain" logic)
COPY mcp_servers/chronicle /app/mcp_servers/chronicle
COPY mcp_servers/protocol /app/mcp_servers/protocol
COPY mcp_servers/task /app/mcp_servers/task
COPY mcp_servers/adr /app/mcp_servers/adr
COPY mcp_servers/agent_persona /app/mcp_servers/agent_persona
COPY mcp_servers/config /app/mcp_servers/config
COPY mcp_servers/code /app/mcp_servers/code

# Copy domain cluster code (The Unified Server entrypoint)
# Source: mcp_servers/gateway/clusters/sanctuary_domain
# Dest: mcp_servers/gateway/clusters/sanctuary_domain
COPY mcp_servers/gateway/clusters/sanctuary_domain /app/mcp_servers/gateway/clusters/sanctuary_domain

ENV PYTHONPATH=/app
ENV PROJECT_ROOT=/app
ENV DOMAIN_SERVER_PORT=8105

# Expose the port
EXPOSE 8105

# Run the Unified Server
CMD ["python", "-m", "mcp_servers.gateway.clusters.sanctuary_domain.server"]

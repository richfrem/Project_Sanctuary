FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install fastmcp

# Copy server code
COPY mcp_servers/document/adr /app/mcp_servers/document/adr

# Set python path
ENV PYTHONPATH=/app
ENV PROJECT_ROOT=/app

# Default command
CMD ["python", "-m", "mcp_servers.document.adr.server"]

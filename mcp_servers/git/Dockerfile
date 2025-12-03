FROM python:3.10-slim

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install fastmcp

# Copy core module
# We need to copy the core module from the parent context
COPY core /app/core

# Copy server code
COPY mcp_servers/system/git_workflow/server.py /app/server.py

# Set python path to include app
ENV PYTHONPATH=/app

# Default command
CMD ["python", "server.py"]

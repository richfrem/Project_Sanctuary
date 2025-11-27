FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create mcp_servers package structure
RUN mkdir -p /app/mcp_servers/task

# Copy Task MCP server code into package structure
COPY __init__.py /app/mcp_servers/task/
COPY models.py /app/mcp_servers/task/
COPY validator.py /app/mcp_servers/task/
COPY operations.py /app/mcp_servers/task/
COPY server.py /app/mcp_servers/task/

# Create mcp_servers __init__.py
RUN touch /app/mcp_servers/__init__.py

# Create directory for task files
RUN mkdir -p /app/TASKS

# Expose MCP server port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app
ENV PYTHONPATH=/app

# Run the MCP server
CMD ["python", "-m", "mcp_servers.task.server"]

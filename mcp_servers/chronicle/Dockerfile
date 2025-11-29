FROM python:3.10-slim

WORKDIR /app

# Install git for git operations if needed (though we use file ops mostly)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app/project_sanctuary

# Set python path
ENV PYTHONPATH=/app/project_sanctuary
ENV PROJECT_ROOT=/app/project_sanctuary

# Run the server
CMD ["python", "-m", "mcp_servers.document.chronicle.server"]

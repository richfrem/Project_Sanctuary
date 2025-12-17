# sanctuary-utils Dockerfile
# Fleet of 7 - Container #1: Low-risk utility tools
# Phase 1: Minimal single-stage (will upgrade to multi-stage in Phase 4)

FROM python:3.11-slim

LABEL maintainer="Project Sanctuary"
LABEL description="Fleet of 7 - sanctuary-utils: Low-risk utility tools"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose SSE endpoint port
EXPOSE 8000

# Health check (Guardrail 3: use proper endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/project_sanctuary

ENV PYTHONPATH=/app/project_sanctuary
ENV PROJECT_ROOT=/app/project_sanctuary

CMD ["python", "-m", "mcp_servers.system.protocol.server"]

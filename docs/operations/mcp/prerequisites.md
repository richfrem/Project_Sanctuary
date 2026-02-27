# Agent Plugin Integration Server Prerequisites

**Last Updated:** 2025-11-26  
**Status:** Canonical

---

## Overview

This document outlines all prerequisites for developing and deploying Agent Plugin Integration (Agent Plugin Integration) servers in Project Sanctuary.

---

## System Requirements

### Operating System
- **macOS** (primary development environment)
- **Linux** (production deployment)
- **Windows** (via WSL2, not primary focus)

### Hardware
- **CPU:** 4+ cores recommended
- **RAM:** 8GB minimum, 16GB recommended
- **Disk:** 20GB free space for containers and images

---

## Required Software

### 1. Podman (Containerization)

**Purpose:** Run Agent Plugin Integration servers in isolated containers

**Installation (macOS):**

```bash
# Option 1: Podman Desktop (Recommended)
# Download from: https://podman-desktop.io/downloads
# Install the .dmg file

# Option 2: Homebrew (CLI only)
brew install podman
```

**Setup:**

```bash
# Initialize Podman machine
podman machine init

# Start Podman machine
podman machine start

# Verify installation
podman --version
# Expected: podman version 5.7.0 (or later)

# Test with hello-world
podman run --rm hello-world
```

**Configuration:**

Add to `~/.zshrc` (if using Homebrew):
```bash
export PATH="/opt/podman/bin:$PATH"
```

Then reload:
```bash
source ~/.zshrc
```

**Verification:**

```bash
# Check machine status
podman machine list
# Should show: Currently running

# Check containers
podman ps
# Should not error

# Run test container
cd tests/podman
./build.sh
# Visit http://localhost:5001 (or 5003)
```

---

### 2. Python 3.11+

**Purpose:** Agent Plugin Integration SDK and server implementation

**Installation:**

```bash
# macOS (Homebrew)
brew install python@3.11

# Verify
python3 --version
# Expected: Python 3.11.x
```

**Virtual Environment:**

```bash
# Create venv for Agent Plugin Integration development
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install Agent Plugin Integration SDK
pip install mcp
```

---

### 3. Agent Plugin Integration SDK

**Purpose:** Agent Plugin Integration implementation

**Installation:**

```bash
# Python SDK
pip install mcp

# Verify
python -c "import mcp; print(mcp.__version__)"
```

**Documentation:**
- [Agent Plugin Integration Specification](https://modelcontextprotocol.io/)
- [Python SDK Docs](https://github.com/modelcontextprotocol/python-sdk)

### 4. Claude Desktop
**Purpose:** Primary interface for interacting with Agent Plugin Integration servers

**Installation:**
- Download from [anthropic.com/claude](https://anthropic.com/claude)

**Configuration:**
- Requires `claude_desktop_config.json` setup (see [[setup_guide|Setup Guide]])

---

## Project-Specific Setup

### 1. Project Sanctuary Repository

```bash
# Clone repository
git clone https://github.com/richfrem/Project_Sanctuary.git
cd Project_Sanctuary

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (for Agent Plugin Integration development)
pip install -r requirements.txt

# For ML/fine-tuning work, use:
# pip install -r requirements-finetuning.txt
```

### 2. Directory Structure

Ensure these directories exist:

```
Project_Sanctuary/
├── tasks/
│   ├── backlog/
│   ├── todo/
│   ├── in-progress/
│   └── done/
├── mcp_servers/
│   └── task/
│       ├── __init__.py
│       ├── models.py
│       ├── validator.py
│       ├── operations.py
│       └── server.py
└── tests/
    └── podman/
```

### 3. Environment Variables

Create `.env` file (if needed):

```bash
# Agent Plugin Integration Server Configuration
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8080

# Project Paths
PROJECT_ROOT=/Users/richardfremmerlid/Projects/Project_Sanctuary
tasks_DIR=${PROJECT_ROOT}/tasks
```

---

## Development Tools (Optional)

### Podman Desktop

**Purpose:** Visual container management

**Features:**
- View running containers
- Monitor resource usage
- View logs
- Start/stop containers
- Port mapping configuration

**Installation:**
Download from https://podman-desktop.io/downloads

**Usage:**
1. Open Podman Desktop
2. Go to **Images** tab to see built images
3. Go to **Containers** tab to manage running containers
4. Click container name to view logs and details

### VS Code Extensions

**Recommended:**
- **Podman** - Container management in VS Code
- **Python** - Python language support
- **Docker** - Dockerfile syntax (works with Podman)

---

## Verification Checklist

Before implementing Agent Plugin Integration servers, verify:

- [ ] Podman installed: `podman --version`
- [ ] Podman machine running: `podman machine list`
- [ ] Can run containers: `podman run --rm hello-world`
- [ ] Python 3.11+ installed: `python3 --version`
- [ ] Agent Plugin Integration SDK installed: `pip show mcp`
- [ ] Test container works: `cd tests/podman && ./build.sh`
- [ ] Can access test page: http://localhost:5001 or 5003
- [ ] Podman Desktop installed (optional but recommended)

---

## Troubleshooting

### Podman Issues

**Problem:** `podman: command not found`

**Solution:**
```bash
# Add to PATH
echo 'export PATH="/opt/podman/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Problem:** `Cannot connect to Podman socket`

**Solution:**
```bash
# Start Podman machine
podman machine start

# Verify
podman machine list
```

**Problem:** Port already in use

**Solution:**
```bash
# Use different port mapping
podman run -p 5003:5001 ...
# Access via http://localhost:5003
```

### Python Issues

**Problem:** `ModuleNotFoundError: No module named 'mcp'`

**Solution:**
```bash
# Activate venv
source .venv/bin/activate

# Install Agent Plugin Integration SDK
pip install mcp
```

---

## Agent Plugin Integration Server-Specific Setup

Once general prerequisites are met, refer to server-specific setup guides:

### RAG Cortex (ChromaDB Vector Database)

The RAG Cortex requires additional setup for the ChromaDB service:

**📖 See: [[SETUP|RAG Cortex Setup Guide]]**

This includes:
- ChromaDB container configuration
- Environment variables (`CHROMA_HOST`, `CHROMA_PORT`)
- Initial database population
- Verification steps

---

## Next Steps

Once all prerequisites are met:

1. ✅ Review [[README|System Architecture]]
2. ✅ Review [[naming_conventions|naming_conventions.md]]
3. ✅ For RAG Cortex: Follow [[SETUP|RAG Cortex Setup Guide]]
4. ✅ For other MCPs: Start with Task #031: Implement Task Agent Plugin Integration
5. Follow implementation tasks #029-#036

---

## References

- [[034_containerize_mcp_servers_with_podman|ADR 034: Containerize Agent Plugin Integration Servers with Podman]]
- [Podman Documentation](https://docs.podman.io/)
- [Agent Plugin Integration Specification](https://modelcontextprotocol.io/)
- [[SETUP|RAG Cortex Setup Guide]]
- [[031_implement_task_mcp|Task #031: Implement Task Agent Plugin Integration]]

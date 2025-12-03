# MCP Server Prerequisites

**Last Updated:** 2025-11-26  
**Status:** Canonical

---

## Overview

This document outlines all prerequisites for developing and deploying MCP (Model Context Protocol) servers in Project Sanctuary.

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

**Purpose:** Run MCP servers in isolated containers

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

**Purpose:** MCP SDK and server implementation

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
# Create venv for MCP development
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install MCP SDK
pip install mcp
```

---

### 3. MCP SDK

**Purpose:** Model Context Protocol implementation

**Installation:**

```bash
# Python SDK
pip install mcp

# Verify
python -c "import mcp; print(mcp.__version__)"
```

**Documentation:**
- [MCP Specification](https://modelcontextprotocol.io/)
- [Python SDK Docs](https://github.com/modelcontextprotocol/python-sdk)

### 4. Claude Desktop
**Purpose:** Primary interface for interacting with MCP servers

**Installation:**
- Download from [anthropic.com/claude](https://anthropic.com/claude)

**Configuration:**
- Requires `claude_desktop_config.json` setup (see [Setup Guide](setup_guide.md))

---

## Project-Specific Setup

### 1. Project Sanctuary Repository

```bash
# Clone repository
git clone https://github.com/richfrem/Project_Sanctuary.git
cd Project_Sanctuary

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (for MCP development)
pip install -r requirements.txt

# For ML/fine-tuning work, use:
# pip install -r requirements-finetuning.txt
```

### 2. Directory Structure

Ensure these directories exist:

```
Project_Sanctuary/
├── TASKS/
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
# MCP Server Configuration
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8080

# Project Paths
PROJECT_ROOT=/Users/richardfremmerlid/Projects/Project_Sanctuary
TASKS_DIR=${PROJECT_ROOT}/TASKS
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

Before implementing MCP servers, verify:

- [ ] Podman installed: `podman --version`
- [ ] Podman machine running: `podman machine list`
- [ ] Can run containers: `podman run --rm hello-world`
- [ ] Python 3.11+ installed: `python3 --version`
- [ ] MCP SDK installed: `pip show mcp`
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

# Install MCP SDK
pip install mcp
```

---

## MCP Server-Specific Setup

Once general prerequisites are met, refer to server-specific setup guides:

### RAG Cortex (ChromaDB Vector Database)

The RAG Cortex requires additional setup for the ChromaDB service:

**📖 See: [RAG Cortex Setup Guide](servers/rag_cortex/SETUP.md)**

This includes:
- ChromaDB container configuration
- Environment variables (`CHROMA_HOST`, `CHROMA_PORT`)
- Initial database population
- Verification steps

---

## Next Steps

Once all prerequisites are met:

1. ✅ Review [architecture.md](./architecture.md)
2. ✅ Review [naming_conventions.md](./naming_conventions.md)
3. ✅ For RAG Cortex: Follow [RAG Cortex Setup Guide](servers/rag_cortex/SETUP.md)
4. ✅ For other MCPs: Start with Task #031: Implement Task MCP
5. Follow implementation tasks #029-#036

---

## References

- [ADR 034: Containerize MCP Servers with Podman](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/ADRs/034_containerize_mcp_servers_with_podman.md)
- [Podman Documentation](https://docs.podman.io/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [RAG Cortex Setup Guide](servers/rag_cortex/SETUP.md)
- [Task #031: Implement Task MCP](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/TASKS/backlog/031_implement_task_mcp.md)

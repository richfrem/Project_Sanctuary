"""
Fleet of 8 - Cluster Definitions

This package contains the containerized MCP server implementations
for the Sanctuary Fleet architecture (ADR 063).

Each subdirectory represents a logical cluster/container:
- sanctuary_utils: Low-risk utility tools
- sanctuary_filesystem: File I/O operations
- sanctuary_network: Network operations
- sanctuary_git: Git workflow tools
- sanctuary_cortex: RAG and semantic search
- sanctuary_domain: Unified domain server (Chronicle, Protocol, Task, ADR, Config)
"""

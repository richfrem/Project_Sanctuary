# Sanctuary Network MCP Server (Fleet Container)

## Overview
The **Sanctuary Network Server** provides external network connectivity and verification tools for the Project Sanctuary system. It allows agents to fetch web content and verify the availability of external resources via HTTP.

## Role
- **Cluster**: Gateway (Internal Network)
- **Container**: `sanctuary_network`
- **Type**: Network Utility Server

## Capabilities

### 1. Fetch URL
- **Tool**: `fetch_url`
- **Purpose**: Retrieves content from a specified URL via HTTP GET.
- **Behavior**: Returns the status code and truncated content (first 2000 characters) for safety.
- **Timeout**: 10 seconds.

### 2. Check Site Status
- **Tool**: `check_site_status`
- **Purpose**: Verifies if a website is reachable and responding.
- **Behavior**: Performs an HTTP HEAD request to check availability without downloading content.
- **Timeout**: 5 seconds.

## Architecture
This server uses `httpx` for asynchronous HTTP requests and exposes its tools via the MCP protocol using `SSEServer`.

### Dependencies
- `httpx` (Async HTTP Client)
- `mcp_servers.lib` (Shared Utilities)

## Usage
The server internally handles network requests and is typically accessed by other agents or the Gateway to perform external validation or data gathering.

# Cluster: sanctuary_network

**Role:** Secure HTTP operations and external connectivity.  
**Port:** 8102  
**Front-end Cluster:** ✅ Yes

## Overview
The `sanctuary_network` cluster acts as the egress gateway for Project Sanctuary, providing audited tools for external URL fetching and site health monitoring.

## Verification Specs (Tier 3: Bridge)
*   **Target:** Gateway Bridge & RPC Routing
*   **Method:** `pytest tests/mcp_servers/gateway/clusters/network/test_gateway.py`

## Tool Inventory & Legacy Mapping
| Function | Gateway Tool Name | Legacy Service | T1/T2 Method |
| :--- | :--- | :--- | :--- |
| **Fetch** | `sanctuary_network-fetch_url` | `rag_cortex` fetch logic | `pytest tests/mcp_servers/network/` |
| **Status** | `sanctuary_network-check_site_status` | `rag_cortex` check logic | |

## Security Profile
- **Default Egress**: Denies all by default.
- **Whitelist**: Managed via Gateway configuration.

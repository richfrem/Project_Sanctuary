# Cluster: sanctuary_utils

**Role:** Primitive utilities and stateless helpers.  
**Port:** 8100  
**Front-end Cluster:** ✅ Yes

## Overview
The `sanctuary_utils` cluster provides a suite of atomic, stateless tools for basic computation, string manipulation, and system metadata.

## Verification Specs (Tier 3: Bridge)
*   **Target:** Gateway Bridge & RPC Routing
*   **Method:** `pytest tests/mcp_servers/gateway/clusters/utils/test_gateway.py`

## Tool Inventory & Legacy Mapping
| Function | Gateway Tool Name | Legacy Operation | T1/T2 Method |
| :--- | :--- | :--- | :--- |
| **Calc** | `sanctuary_utils-calculator-calculate` | `calculate` | `pytest tests/mcp_servers/utils/` |
| **Add** | `sanctuary_utils-calculator-add` | `add` | |
| **Sub** | `sanctuary_utils-calculator-subtract` | `subtract` | |
| **Mul** | `sanctuary_utils-calculator-multiply` | `multiply` | |
| **Div** | `sanctuary_utils-calculator-divide` | `divide` | |
| **Time** | `sanctuary_utils-time-get-current-time` | `get_current_time` | |
| **TZ** | `sanctuary_utils-time-get-timezone-info` | `get_timezone_info`| |
| **UUID4** | `sanctuary_utils-uuid-generate-uuid4` | `generate_uuid4` | |
| **UUID1** | `sanctuary_utils-uuid-generate-uuid1` | `generate_uuid1` | |
| **UUID Check**| `sanctuary_utils-uuid-validate-uuid` | `validate_uuid` | |
| **Upper** | `sanctuary_utils-string-to-upper` | `to_upper` | |
| **Lower** | `sanctuary_utils-string-to-lower` | `to_lower` | |
| **Trim** | `sanctuary_utils-string-trim` | `trim` | |
| **Reverse** | `sanctuary_utils-string-reverse` | `reverse` | |
| **Count** | `sanctuary_utils-string-word-count` | `word_count` | |
| **Replace** | `sanctuary_utils-string-replace` | `replace` | |

## Logic Notes
All tools in this cluster are stateless and do not persist data to the Sanctuary filesystem.

#============================================
# mcp_servers/gateway/clusters/sanctuary_utils/tools/time_tool.py
# Purpose: Provides current time and timezone information.
# Role: Functional Tool
#============================================
from datetime import datetime, timezone
from typing import Any


#============================================
# Function: get_current_time
# Purpose: Get the current time in the specified timezone.
# Args:
#   timezone_name: Timezone name (default: UTC).
# Returns: Dictionary with current time information.
#============================================
def get_current_time(timezone_name: str = "UTC") -> dict[str, Any]:
    try:
        now = datetime.now(timezone.utc)
        return {
            "success": True,
            "time": now.isoformat(),
            "timezone": timezone_name,
            "unix_timestamp": int(now.timestamp()),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


#============================================
# Function: get_timezone_info
# Purpose: Get information about available timezones.
# Returns: Dictionary with timezone information.
#============================================
def get_timezone_info() -> dict[str, Any]:
    return {
        "success": True,
        "available_timezones": ["UTC"],
        "note": "Currently only UTC is supported. Future versions will support pytz timezones.",
    }


# Tool manifest for registration
TOOL_MANIFEST = {
    "name": "time",
    "description": "Get current time and timezone information",
    "functions": [
        {
            "name": "get_current_time",
            "description": "Get the current time in UTC or specified timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_name": {
                        "type": "string",
                        "description": "Timezone name (default: UTC)",
                        "default": "UTC",
                    }
                },
            },
        },
        {
            "name": "get_timezone_info",
            "description": "Get information about available timezones",
            "parameters": {"type": "object", "properties": {}},
        },
    ],
}

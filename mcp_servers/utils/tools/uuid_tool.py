"""
UUID Tool for sanctuary-utils
Provides UUID generation utilities.
"""
import uuid
from typing import Any


def generate_uuid4() -> dict[str, Any]:
    """Generate a random UUID (version 4)."""
    try:
        return {
            "success": True,
            "uuid": str(uuid.uuid4()),
            "version": 4,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_uuid1() -> dict[str, Any]:
    """Generate a UUID based on host ID and current time (version 1)."""
    try:
        return {
            "success": True,
            "uuid": str(uuid.uuid1()),
            "version": 1,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def validate_uuid(uuid_string: str) -> dict[str, Any]:
    """Validate if a string is a valid UUID."""
    try:
        parsed = uuid.UUID(uuid_string)
        return {
            "success": True,
            "valid": True,
            "uuid": str(parsed),
            "version": parsed.version,
        }
    except ValueError:
        return {
            "success": True,
            "valid": False,
            "error": "Invalid UUID format",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Tool manifest for registration
TOOL_MANIFEST = {
    "name": "uuid",
    "description": "UUID generation and validation utilities",
    "functions": [
        {
            "name": "generate_uuid4",
            "description": "Generate a random UUID (version 4)",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "generate_uuid1",
            "description": "Generate a UUID based on host ID and time (version 1)",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "validate_uuid",
            "description": "Validate if a string is a valid UUID",
            "parameters": {
                "type": "object",
                "properties": {
                    "uuid_string": {
                        "type": "string",
                        "description": "UUID string to validate",
                    }
                },
                "required": ["uuid_string"],
            },
        },
    ],
}

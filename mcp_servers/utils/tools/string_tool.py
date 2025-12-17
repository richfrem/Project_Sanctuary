"""
String Tool for sanctuary-utils
Provides string manipulation utilities.
"""
from typing import Any


def to_upper(text: str) -> dict[str, Any]:
    """Convert text to uppercase."""
    try:
        return {"success": True, "result": text.upper()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def to_lower(text: str) -> dict[str, Any]:
    """Convert text to lowercase."""
    try:
        return {"success": True, "result": text.lower()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def trim(text: str) -> dict[str, Any]:
    """Remove leading and trailing whitespace."""
    try:
        return {"success": True, "result": text.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def reverse(text: str) -> dict[str, Any]:
    """Reverse a string."""
    try:
        return {"success": True, "result": text[::-1]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def word_count(text: str) -> dict[str, Any]:
    """Count words in text."""
    try:
        words = text.split()
        return {
            "success": True,
            "word_count": len(words),
            "char_count": len(text),
            "char_count_no_spaces": len(text.replace(" ", "")),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def replace(text: str, old: str, new: str) -> dict[str, Any]:
    """Replace occurrences of old with new in text."""
    try:
        result = text.replace(old, new)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Tool manifest for registration
TOOL_MANIFEST = {
    "name": "string",
    "description": "String manipulation utilities",
    "functions": [
        {
            "name": "to_upper",
            "description": "Convert text to uppercase",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to convert"}},
                "required": ["text"],
            },
        },
        {
            "name": "to_lower",
            "description": "Convert text to lowercase",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to convert"}},
                "required": ["text"],
            },
        },
        {
            "name": "trim",
            "description": "Remove leading and trailing whitespace",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to trim"}},
                "required": ["text"],
            },
        },
        {
            "name": "reverse",
            "description": "Reverse a string",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to reverse"}},
                "required": ["text"],
            },
        },
        {
            "name": "word_count",
            "description": "Count words and characters in text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to analyze"}},
                "required": ["text"],
            },
        },
        {
            "name": "replace",
            "description": "Replace occurrences in text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Original text"},
                    "old": {"type": "string", "description": "Substring to replace"},
                    "new": {"type": "string", "description": "Replacement substring"},
                },
                "required": ["text", "old", "new"],
            },
        },
    ],
}

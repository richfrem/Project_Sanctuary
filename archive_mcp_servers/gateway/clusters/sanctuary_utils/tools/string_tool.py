#============================================
# mcp_servers/gateway/clusters/sanctuary_utils/tools/string_tool.py
# Purpose: String manipulation utilities.
# Role: Functional Tool
#============================================
from typing import Any


#============================================
# Function: to_upper
# Purpose: Convert text to uppercase.
#============================================
def to_upper(text: str) -> dict[str, Any]:
    try:
        return {"success": True, "result": text.upper()}
    except Exception as e:
        return {"success": False, "error": str(e)}


#============================================
# Function: to_lower
# Purpose: Convert text to lowercase.
#============================================
def to_lower(text: str) -> dict[str, Any]:
    try:
        return {"success": True, "result": text.lower()}
    except Exception as e:
        return {"success": False, "error": str(e)}


#============================================
# Function: trim
# Purpose: Remove leading and trailing whitespace.
#============================================
def trim(text: str) -> dict[str, Any]:
    try:
        return {"success": True, "result": text.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}


#============================================
# Function: reverse
# Purpose: Reverse a string.
#============================================
def reverse(text: str) -> dict[str, Any]:
    try:
        return {"success": True, "result": text[::-1]}
    except Exception as e:
        return {"success": False, "error": str(e)}


#============================================
# Function: word_count
# Purpose: Count words in text.
#============================================
def word_count(text: str) -> dict[str, Any]:
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


#============================================
# Function: replace
# Purpose: Replace occurrences of old with new in text.
#============================================
def replace(text: str, old: str, new: str) -> dict[str, Any]:
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

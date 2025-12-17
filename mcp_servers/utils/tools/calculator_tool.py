"""
Calculator Tool for sanctuary-utils
Provides basic math operations with fault containment.
"""
from typing import Any, Union
import math


def calculate(expression: str) -> dict[str, Any]:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: Math expression (e.g., "2 + 2", "sqrt(16)", "10 * 5")
        
    Returns:
        Dictionary with result or error.
    """
    try:
        # Safe evaluation - only allow math operations
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "pi": math.pi,
            "e": math.e,
        }
        
        # Parse and evaluate safely
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        return {
            "success": True,
            "expression": expression,
            "result": result,
        }
    except ZeroDivisionError:
        return {
            "success": False,
            "expression": expression,
            "error": "Division by zero",
        }
    except Exception as e:
        return {
            "success": False,
            "expression": expression,
            "error": str(e),
        }


def add(a: Union[int, float], b: Union[int, float]) -> dict[str, Any]:
    """Add two numbers."""
    try:
        return {"success": True, "result": a + b}
    except Exception as e:
        return {"success": False, "error": str(e)}


def subtract(a: Union[int, float], b: Union[int, float]) -> dict[str, Any]:
    """Subtract b from a."""
    try:
        return {"success": True, "result": a - b}
    except Exception as e:
        return {"success": False, "error": str(e)}


def multiply(a: Union[int, float], b: Union[int, float]) -> dict[str, Any]:
    """Multiply two numbers."""
    try:
        return {"success": True, "result": a * b}
    except Exception as e:
        return {"success": False, "error": str(e)}


def divide(a: Union[int, float], b: Union[int, float]) -> dict[str, Any]:
    """Divide a by b."""
    try:
        if b == 0:
            return {"success": False, "error": "Division by zero"}
        return {"success": True, "result": a / b}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Tool manifest for registration
TOOL_MANIFEST = {
    "name": "calculator",
    "description": "Basic math operations and expression evaluation",
    "functions": [
        {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '2 + 2', 'sqrt(16)')",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "subtract",
            "description": "Subtract b from a",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Number to subtract"},
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "divide",
            "description": "Divide a by b",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Dividend"},
                    "b": {"type": "number", "description": "Divisor"},
                },
                "required": ["a", "b"],
            },
        },
    ],
}

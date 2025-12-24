#============================================
# mcp_servers/gateway/clusters/sanctuary_utils/tools/calculator_tool.py
# Purpose: Basic math operations with fault containment.
# Role: Functional Tool
#============================================
from typing import Any, Union
import math


#============================================
# Function: calculate
# Purpose: Evaluate a mathematical expression safely.
# Args:
#   expression: Math expression (e.g., "2 + 2", "sqrt(16)")
# Returns: Dictionary with result or error
#============================================
def calculate(expression: str) -> dict[str, Any]:
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


#============================================
# Function: add
# Purpose: Add two numbers.
#============================================
def add(a: Union[int, float], b: Union[int, float]) -> dict[str, Any]:
    try:
        return {"success": True, "result": a + b}
    except Exception as e:
        return {"success": False, "error": str(e)}


#============================================
# Function: subtract
# Purpose: Subtract b from a.
#============================================
def subtract(a: Union[int, float], b: Union[int, float]) -> dict[str, Any]:
    try:
        return {"success": True, "result": a - b}
    except Exception as e:
        return {"success": False, "error": str(e)}


#============================================
# Function: multiply
# Purpose: Multiply two numbers.
#============================================
def multiply(a: Union[int, float], b: Union[int, float]) -> dict[str, Any]:
    try:
        return {"success": True, "result": a * b}
    except Exception as e:
        return {"success": False, "error": str(e)}


#============================================
# Function: divide
# Purpose: Divide a by b.
#============================================
def divide(a: Union[int, float], b: Union[int, float]) -> dict[str, Any]:
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

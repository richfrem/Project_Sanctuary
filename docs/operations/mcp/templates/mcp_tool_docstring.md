# Agent Plugin Integration Tool Docstring Template

Use this template for all functions decorated with `@mcp.tool()`.

```python
@mcp.tool()
def tool_name(arg1: str, arg2: int = 10) -> str:
    """
    [One-line summary of the tool's purpose.]

    [Detailed description of the tool's functionality, including any
    side effects, external API calls, or important constraints.]

    Args:
        arg1: [Description of arg1]
        arg2: [Description of arg2]. Defaults to 10.

    Returns:
        [Description of the return value. If it returns a JSON string,
        describe the schema.]

    Raises:
        ValueError: [Condition under which this error is raised]
        RuntimeError: [Condition under which this error is raised]
    """
    # Implementation
```

## Example

```python
@mcp.tool()
def calculate_metric(data: List[float], metric_type: str = "mean") -> float:
    """
    Calculates a statistical metric for a given dataset.

    Supported metrics are 'mean', 'median', and 'std_dev'. This tool
    is used by the Analyst Persona for data processing.

    Args:
        data: A list of floating-point numbers to analyze.
        metric_type: The type of metric to calculate. Options: "mean",
            "median", "std_dev". Defaults to "mean".

    Returns:
        The calculated metric as a float.

    Raises:
        ValueError: If an unsupported metric_type is provided or if
            data is empty.
    """
    # ...
```

# Evolution MCP Server (Protocol 131)

**Description:** The Evolution MCP Server implements **Protocol 131 (Evolutionary Self-Improvement)**. It provides the metric calculation engine for the **Map-Elites** Quality-Diversity algorithm, allowing the system to objectively evaluate and evolve its own prompts and protocols.

## Core Responsibilities

1.  **Metric Calculation:** Computes "Depth" and "Scope" scores for textual content.
2.  **Fitness Evaluation:** Provides the objective function for the evolutionary search.

## Tools

| Tool Name | Description | Protocol |
|-----------|-------------|----------|
| `measure_fitness` | Returns a full fitness vector (`{depth, scope}`) for a given text. | P131 |
| `evaluate_depth` | Calculates **Depth (0.0-5.0)**: Based on citation density and technical complexity. | P131 |
| `evaluate_scope` | Calculates **Scope (0.0-5.0)**: Based on file touch width and domain breadth. | P131 |

## Map-Elites Dimensions

- **Depth (Y-Axis):** Measures rigor. High depth means dense citations and high technical specificity.
- **Scope (X-Axis):** Measures breadth. High scope means the content bridges multiple architectural domains.

## Configuration

### MCP Config
```json
"evolution": {
  "command": "uv",
  "args": ["run", "mcp_servers/evolution/server.py"],
  "env": { "PROJECT_ROOT": "..." }
}
```

## Testing

Run the dedicated test suite:
```bash
pytest tests/mcp_servers/evolution/
```

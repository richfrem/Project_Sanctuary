Analyze this CLI tool source code and extract a comprehensive JSON summary.
The summary must map to the project's "Gold Standard" header format.

PRIORITY: If the file contains a top-level docstring header (Gold Standard), extract details DIRECTLY from there.

File: {file_path}
Code:
{content}

Instructions:
1. "purpose": Extract from the "Purpose" section of the header.
2. "layer": Extract from "Layer".
3. "supported_object_types": Extract from "Supported Object Types" (if present).
4. "usage": Extract from "Usage" or "Usage Examples".
5. "args": Extract from "CLI Arguments".
6. "inputs": Extract from "Input Files".
7. "outputs": Extract from "Output".
8. "dependencies": Extract from "Script Dependencies".
9. "consumed_by": Extract from "Consumed by".
10. "key_functions": Extract from "Key Functions".
11. If header is missing, infer these fields from the code.

Format (Strict JSON, no markdown code blocks if possible, just the JSON):
{
  "purpose": "...",
  "layer": "...",
  "supported_object_types": ["..."],
  "usage": ["..."],
  "args": ["..."],
  "inputs": ["..."],
  "outputs": ["..."],
  "dependencies": ["..."],
  "consumed_by": ["..."],
  "key_functions": ["..."]
}

JSON:

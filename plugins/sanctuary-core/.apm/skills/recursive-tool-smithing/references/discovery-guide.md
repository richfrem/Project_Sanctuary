# Discovery & Bootstrap Guide

Guide for setting up an autonomous tool evolution loop in Bootstrap Mode.

## Bootstrap Checklist
1. **Run tag**: Identifier for the run (e.g., `nov01`). Work occurs on branch `autoresearch/<tag>`.
2. **Project scope**: Target capability area (e.g., Project Sanctuary).
3. **Self-improvement goal**: Metric to optimize (e.g. latency, memory efficiency).
4. **Success threshold**: Minimum fitness score improvement to keep candidate (default: 10%).
5. **Safety boundaries**: Restrict candidate generation to `scripts/toolbox/`.

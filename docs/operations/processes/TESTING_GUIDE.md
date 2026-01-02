# Testing Guide

**Status:** Active  
**Last Updated:** 2026-01-02

## Overview

Project Sanctuary uses a 3-layer test pyramid for all MCP servers.

## Quick Links

| Resource | Location |
|----------|----------|
| **Main Test Suite README** | [tests/README.md](../tests/README.md) |
| **MCP Testing Standards** | [TESTING_STANDARDS.md](../mcp/TESTING_STANDARDS.md) |
| **Test Pyramid Diagram** | [mcp_test_pyramid.mmd](../../architecture_diagrams/system/mcp_test_pyramid.mmd) |

## Test Pyramid

| Layer | Purpose | Speed | Command |
|-------|---------|-------|---------|
| **Unit** | Isolated logic | Fast (ms) | `pytest tests/mcp_servers/<server>/unit/` |
| **Integration** | Real local services | Medium (sec) | `pytest tests/mcp_servers/<server>/integration/` |
| **E2E** | Full MCP protocol | Slow (min) | `pytest -m e2e` |

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific server
pytest tests/mcp_servers/git/ -v

# With coverage
pytest tests/ --cov=mcp_servers --cov-report=html

# Headless E2E (CI-friendly)
pytest tests/mcp_servers/ -m headless -v
```

## Related Documents

- [ADR 053](../ADRs/053_mcp_test_pyramid.md) - Test Pyramid Architecture
- [Protocol 101](../../../01_PROTOCOLS/101_The_Doctrine_of_the_Unbreakable_Commit.md) - Tests must pass before commit

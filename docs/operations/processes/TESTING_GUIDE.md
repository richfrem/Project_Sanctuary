# Testing Guide

**Status:** Active  
**Last Updated:** 2026-01-02

## Overview

Project Sanctuary uses a 3-layer test pyramid for all Agent Plugin Integration servers.

## Quick Links

| Resource | Location |
|----------|----------|
| **Main Test Suite README** | [[README|tests/README.md]] |
| **Agent Plugin Integration Testing Standards** | [[TESTING_STANDARDS|TESTING_STANDARDS.md]] |
| **Test Pyramid Diagram** | [[mcp_test_pyramid.mmd|mcp_test_pyramid.mmd]] |

## Test Pyramid

| Layer | Purpose | Speed | Command |
|-------|---------|-------|---------|
| **Unit** | Isolated logic | Fast (ms) | `pytest tests/mcp_servers/<server>/unit/` |
| **Integration** | Real local services | Medium (sec) | `pytest tests/mcp_servers/<server>/integration/` |
| **E2E** | Full Agent Plugin Integration protocol | Slow (min) | `pytest -m e2e` |

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

- [[053_standardize_live_integration_testing_pattern|ADR 053]] - Test Pyramid Architecture
- [[101_The_Doctrine_of_the_Unbreakable_Commit|Protocol 101]] - Tests must pass before commit

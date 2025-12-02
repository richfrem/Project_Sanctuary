# Task 021C: MCP Integration & Performance Test Suite

## Metadata
- **Status**: Done
- **Priority**: medium
- **Complexity**: medium
- **Category**: testing
- **Estimated Effort**: 4-6 hours
- **Dependencies**: 021A, 021B, 086
- **Assigned To**: Antigravity
- **Created**: 2025-11-21
- **Updated**: 2025-12-01 (Refactored for MCP Architecture)
- **Parent Task**: 021 (split into 021A, 021B, 021C)

## Context

With the migration to 11 distributed MCP servers, integration testing must verify the communication *between* these servers (e.g., Council -> Cortex, Council -> Git) rather than just internal module calls. Performance benchmarks should measure the latency of MCP tool calls.

## Objective

Create and maintain a robust integration test suite for cross-MCP workflows and establish performance baselines for critical MCP tools.

## Acceptance Criteria

### 1. MCP Integration Tests
- [x] Create `tests/integration/test_agent_persona_with_cortex.py` ✅
  - Verifies Agent Persona -> Cortex context retrieval
  - Verifies multi-agent deliberation with context
  - **Status**: PASSING (Task #086A)
- [x] Create `tests/integration/test_council_with_git.py`
  - Verify Council can trigger Git operations via Git MCP
  - Test end-to-end "Plan -> Code -> Commit" flow
- [x] Create `tests/integration/test_chronicle_integration.py`
  - Verify other MCPs can write to Chronicle
- [x] Create `tests/integration/test_forge_model_serving.py`
  - Verify Forge MCP correctly serves models to Agent Persona

### 2. Performance Benchmarks
- [x] Create `tests/benchmarks/test_mcp_tool_latency.py`
  - Benchmark `cortex_query` latency
  - Benchmark `agent_persona_dispatch` latency
  - Benchmark `git_status` latency
- [x] Add pytest-benchmark configuration ✅
  - Installed pytest-benchmark
  - Configured in pytest.ini
- [x] Create performance baseline report

### 3. Test Organization
- [x] Mark integration tests with `@pytest.mark.integration` ✅
- [x] Mark performance tests with `@pytest.mark.benchmark` ✅
- [x] Configure pytest to skip slow tests by default ✅
- [x] Add `run_integration_tests.sh` script ✅

## Technical Approach

```python
# tests/integration/test_council_with_git.py
import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.lib.council.council_ops import CouncilOperations

@pytest.mark.integration
def test_council_git_flow():
    """Test Council directing Git operations."""
    # Mock Git MCP client
    with patch('mcp_servers.lib.council.council_ops.get_mcp_client') as mock_client:
        council = CouncilOperations()
        
        # Simulate task that requires git commit
        result = council.dispatch_task(
            "Create a feature branch for new protocol",
            agent="coordinator"
        )
        
        # Verify Git MCP tool was called
        mock_client.call_tool.assert_called_with(
            "git_create_branch", 
            {"branch_name": "feat/new-protocol"}
        )
```

## Success Metrics

- [ ] Critical cross-MCP flows have integration tests
- [ ] Integration tests pass consistently in CI
- [ ] Performance baselines established for top 5 most used tools
- [ ] Clear documentation for running integration tests

## Current Status (2025-12-01)

- **Agent Persona <-> Cortex:** ✅ Fully tested and passing (Task #086A)
- **Legacy Tests:** `test_rag_simple.py` and others need evaluation for porting to MCP architecture or archiving.

## Related Protocols

- **P89**: The Clean Forge
- **P101**: The Unbreakable Commit
- **P115**: The Tactical Mandate

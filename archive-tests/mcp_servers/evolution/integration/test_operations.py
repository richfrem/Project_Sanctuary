"""
Evolution MCP Integration Tests - Operations Testing
====================================================

Comprehensive integration tests for all Evolution operations (Protocol 131).
Uses BaseIntegrationTest and follows the pattern in rag_cortex/integration/test_operations.py.

MCP OPERATIONS:
---------------
| Operation        | Type | Description                              |
|------------------|------|------------------------------------------|
| calculate_fitness| READ | Calculates Depth and Scope metrics       |
| measure_depth    | READ | Calculates Depth metric (0-5)            |
| measure_scope    | READ | Calculates Scope metric (0-5)            |
"""
import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest
from mcp_servers.evolution.operations import EvolutionOperations

class TestEvolutionOperations(BaseIntegrationTest):
    """
    Integration tests for all Evolution operations.
    Follows Protocol 131 metric logic.
    """

    def get_required_services(self):
        """No external services required for Evolution logic."""
        return []

    @pytest.fixture
    def evolution_ops(self, tmp_path):
        project_root = tmp_path / "project_root"
        project_root.mkdir()
        
        # Setup structure
        (project_root / "00_CHRONICLE").mkdir()
        (project_root / "01_PROTOCOLS").mkdir()
        
        ops = EvolutionOperations(str(project_root))
        return ops

    #===========================================================================
    # MCP OPERATION: calculate_fitness
    #===========================================================================
    def test_calculate_fitness(self, evolution_ops):
        """Verify complex fitness calculation across multiple dimensions."""
        content = "This is a technical doc with citations [1](http://example.com) and code `src/main.py`."
        
        result = evolution_ops.calculate_fitness(content)
        assert "depth" in result
        assert "scope" in result
        assert result["depth"] > 0
        assert result["scope"] > 0

    #===========================================================================
    # MCP OPERATION: measure_depth
    #===========================================================================
    def test_measure_depth(self, evolution_ops):
        """Verify Depth metric calculation (0-5 scale)."""
        content = "Simple content."
        score = evolution_ops.measure_depth(content)
        assert 0 <= score <= 5

    #===========================================================================
    # MCP OPERATION: measure_scope
    #===========================================================================
    def test_measure_scope(self, evolution_ops):
        """Verify Scope metric calculation (0-5 scale)."""
        content = "Touching `ADRs/001.md` and `scripts/sync.py`."
        score = evolution_ops.measure_scope(content)
        assert 0 <= score <= 5

    #===========================================================================
    # EDGE CASES
    #===========================================================================
    def test_empty_content(self, evolution_ops):
        """Verify scores for empty content."""
        result = evolution_ops.calculate_fitness("")
        assert result["depth"] == 0.0
        assert result["scope"] == 0.0
        
        result = evolution_ops.calculate_fitness("   ")
        assert result["depth"] == 0.0
        assert result["scope"] == 0.0

    def test_high_complexity_content(self, evolution_ops):
        """Verify depth scores for technical content."""
        # Content with many citations and long words
        content = (
            "The implementation utilizes asynchronous coroutines for high-performance I/O multiplexing. "
            "See [docs](http://example.com/api) and [spec](http://example.com/rfc). "
            "Internal references like `mcp_servers/lib/sse_adaptor.py` and `mcp_servers/evolution/server.py` "
            "demonstrate architectural breadth."
        )
        result = evolution_ops.calculate_fitness(content)
        assert result["scope"] > 1.0

    def test_measure_scope_path_filtering(self, evolution_ops):
        """Verify internal paths starting with 'p', 't', 'h' are NOT excluded."""
        content = "Check [Protocol](protocols/P128.md) and [Task](tasks/T01.md)."
        score = evolution_ops.measure_scope(content)
        # Should detect 2 distinct domains (protocols, tasks) -> High score
        assert score >= 1.0 

"""
Tests for Protocol 87 MCP Orchestrator

Tests the structured query routing to specialized MCPs.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.cognitive.cortex.operations import CortexOperations
from mcp_servers.cognitive.cortex.mcp_client import MCPClient


class TestProtocol87Orchestrator:
    """Test Protocol 87 structured query orchestration."""
    
    @pytest.fixture
    def ops(self, tmp_path):
        """Create CortexOperations instance."""
        return CortexOperations(str(project_root))
    
    @pytest.fixture
    def mcp_client(self):
        """Create MCPClient instance."""
        return MCPClient(str(project_root))
    
    def test_parse_protocol_query(self, ops):
        """Test parsing Protocol 87 query string."""
        from mcp_servers.cognitive.cortex.structured_query import parse_query_string
        
        query = 'RETRIEVE :: Protocols :: Name="Protocol 101"'
        result = parse_query_string(query)
        
        assert result["intent"] == "RETRIEVE"
        assert result["scope"] == "Protocols"
        assert "Protocol 101" in result["constraints"]
    
    def test_route_to_protocol_mcp(self, mcp_client):
        """Test routing to Protocol MCP."""
        results = mcp_client.route_query(
            scope="Protocols",
            intent="RETRIEVE",
            constraints='Name="Protocol 101"',
            query_data={}
        )
        
        assert len(results) > 0
        assert results[0]["source"] == "Protocol MCP"
        assert results[0]["mcp_tool"] == "protocol_get"
    
    def test_route_to_chronicle_mcp(self, mcp_client):
        """Test routing to Chronicle MCP."""
        results = mcp_client.route_query(
            scope="Living_Chronicle",
            intent="SUMMARIZE",
            constraints="Timeframe=Recent",
            query_data={}
        )
        
        assert len(results) > 0
        assert results[0]["source"] == "Chronicle MCP"
        assert results[0]["mcp_tool"] == "chronicle_list_entries"
    
    def test_route_to_task_mcp(self, mcp_client):
        """Test routing to Task MCP."""
        results = mcp_client.route_query(
            scope="Tasks",
            intent="SUMMARIZE",
            constraints='Status="in-progress"',
            query_data={}
        )
        
        assert len(results) > 0
        assert results[0]["source"] == "Task MCP"
        assert results[0]["mcp_tool"] == "list_tasks"
    
    def test_route_to_adr_mcp(self, mcp_client):
        """Test routing to ADR MCP."""
        results = mcp_client.route_query(
            scope="ADRs",
            intent="SUMMARIZE",
            constraints="",
            query_data={}
        )
        
        assert len(results) > 0
        assert results[0]["source"] == "ADR MCP"
        assert results[0]["mcp_tool"] == "adr_list"
    
    def test_query_structured_protocol(self, ops):
        """Test structured query for protocols."""
        result = ops.query_structured('RETRIEVE :: Protocols :: Name="Protocol 101"')
        
        assert result["request_id"]
        assert result["steward_id"] == "CORTEX-MCP-01"
        assert "routing" in result
        assert result["routing"]["scope"] == "Protocols"
        assert result["routing"]["routed_to"] == "Protocol MCP"
    
    def test_query_structured_chronicle(self, ops):
        """Test structured query for chronicles."""
        result = ops.query_structured("SUMMARIZE :: Living_Chronicle :: Timeframe=Recent")
        
        assert result["request_id"]
        assert "routing" in result
        assert result["routing"]["scope"] == "Living_Chronicle"
        assert result["routing"]["routed_to"] == "Chronicle MCP"
    
    def test_query_structured_with_request_id(self, ops):
        """Test structured query with custom request ID."""
        custom_id = "test-request-123"
        result = ops.query_structured(
            'RETRIEVE :: Protocols :: Name="Protocol 101"',
            request_id=custom_id
        )
        
        assert result["request_id"] == custom_id
    
    def test_query_structured_error_handling(self, ops):
        """Test error handling for malformed queries."""
        result = ops.query_structured("INVALID QUERY FORMAT")
        
        assert result["status"] == "error"
        assert "error" in result
    
    def test_mcp_name_mapping(self, ops):
        """Test MCP name mapping."""
        assert ops._get_mcp_name("Protocols") == "Protocol MCP"
        assert ops._get_mcp_name("Living_Chronicle") == "Chronicle MCP"
        assert ops._get_mcp_name("Tasks") == "Task MCP"
        assert ops._get_mcp_name("Code") == "Code MCP"
        assert ops._get_mcp_name("ADRs") == "ADR MCP"
        assert ops._get_mcp_name("Unknown") == "Cortex MCP (Vector DB)"


@pytest.mark.integration
class TestProtocol87Integration:
    """Integration tests for Protocol 87 orchestration."""
    
    @pytest.fixture
    def ops(self):
        """Create CortexOperations with real project root."""
        return CortexOperations(str(project_root))
    
    def test_end_to_end_protocol_query(self, ops):
        """Test end-to-end protocol query."""
        result = ops.query_structured('RETRIEVE :: Protocols :: Name="Protocol 101"')
        
        # Verify response structure
        assert "request_id" in result
        assert "steward_id" in result
        assert "timestamp_utc" in result
        assert "matches" in result
        assert "routing" in result
        
        # Verify routing
        assert result["routing"]["scope"] == "Protocols"
        assert result["routing"]["orchestrator"] == "CORTEX-MCP-01"
    
    def test_end_to_end_chronicle_query(self, ops):
        """Test end-to-end chronicle query."""
        result = ops.query_structured("SUMMARIZE :: Living_Chronicle :: Timeframe=Recent")
        
        assert result["routing"]["scope"] == "Living_Chronicle"
        assert "matches" in result
    
    def test_cross_mcp_capability(self, ops):
        """Test that different scopes route to different MCPs."""
        # Query protocols
        protocol_result = ops.query_structured('RETRIEVE :: Protocols :: Name="Protocol 101"')
        
        # Query chronicles
        chronicle_result = ops.query_structured("SUMMARIZE :: Living_Chronicle :: Timeframe=Recent")
        
        # Verify different routing
        assert protocol_result["routing"]["routed_to"] == "Protocol MCP"
        assert chronicle_result["routing"]["routed_to"] == "Chronicle MCP"

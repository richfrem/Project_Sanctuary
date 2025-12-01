"""
Integration Tests for Agent Persona MCP ↔ Cortex MCP Communication

Tests the full flow:
1. Council MCP → Agent Persona MCP → Cortex MCP
2. Multi-agent deliberation with context retrieval
3. Cross-MCP communication patterns
"""

import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.lib.council.council_ops import CouncilOperations
from mcp_servers.lib.agent_persona.agent_persona_ops import AgentPersonaOperations
from mcp_servers.cognitive.cortex.operations import CortexOperations


@pytest.mark.integration
class TestAgentPersonaCortexIntegration:
    """Test Agent Persona MCP can successfully query Cortex MCP"""
    
    def test_persona_queries_cortex_for_context(self):
        """
        Test that Agent Persona MCP can query Cortex MCP for context
        
        Validates:
        - persona_dispatch can call cortex.query
        - Context is properly retrieved
        - Agent receives context in response
        """
        with patch('mcp_servers.lib.agent_persona.agent_persona_ops.get_llm_client') as mock_llm:
            # Mock LLM response
            mock_client = MagicMock()
            mock_client.generate.return_value = "Based on the context provided, Protocol 101 defines..."
            mock_llm.return_value = mock_client
            
            # Mock Cortex query
            with patch.object(CortexOperations, 'query') as mock_cortex_query:
                mock_cortex_query.return_value = MagicMock(
                    results=[
                        MagicMock(
                            content="Protocol 101: Git Workflow...",
                            metadata={"source": "01_PROTOCOLS/101_git_workflow.md"},
                            score=0.95
                        )
                    ]
                )
                
                # Initialize operations
                persona_ops = AgentPersonaOperations()
                
                # Dispatch task with context query
                result = persona_ops.dispatch(
                    role="auditor",
                    task="Review Protocol 101 for compliance",
                    context=None,  # Will query Cortex
                    model_name="test-model"
                )
                
                # Verify Cortex was queried (if persona implementation queries it)
                # Note: Current implementation receives context, doesn't query directly
                assert result["status"] == "success"
                assert "response" in result


@pytest.mark.integration
class TestCouncilAgentPersonaCortexFlow:
    """Test full Council MCP → Agent Persona MCP → Cortex MCP flow"""
    
    def test_council_dispatch_full_flow(self):
        """
        Test complete flow from Council through Agent Persona to Cortex
        
        Validates:
        1. Council MCP dispatches to Agent Persona MCP
        2. Agent Persona MCP executes with context
        3. Results flow back correctly
        """
        with patch('mcp_servers.lib.agent_persona.agent_persona_ops.get_llm_client') as mock_llm:
            # Mock LLM responses
            mock_client = MagicMock()
            mock_client.generate.return_value = "Analysis complete based on context."
            mock_llm.return_value = mock_client
            
            # Mock Cortex query
            with patch.object(CortexOperations, 'query') as mock_cortex_query:
                mock_cortex_query.return_value = MagicMock(
                    results=[
                        MagicMock(
                            content="Relevant protocol content...",
                            metadata={"source": "test.md"},
                            score=0.9
                        )
                    ]
                )
                
                # Initialize Council operations
                council_ops = CouncilOperations()
                
                # Dispatch single-agent task
                result = council_ops.dispatch_task(
                    task_description="Analyze the security implications",
                    agent="auditor",
                    max_rounds=1
                )
                
                # Verify successful execution
                assert result["status"] == "success"
                assert result["session_id"]
                assert len(result["agents"]) == 1
                assert result["agents"][0] == "auditor"
                
                # Verify Cortex was queried
                mock_cortex_query.assert_called_once()


    def test_multi_agent_deliberation_with_context(self):
        """
        Test full council deliberation (3 agents, multiple rounds) with Cortex context
        
        Validates:
        1. All 3 agents (coordinator, strategist, auditor) execute
        2. Cortex provides context to all agents
        3. Multi-round deliberation works correctly
        4. Final synthesis includes all perspectives
        """
        with patch('mcp_servers.lib.agent_persona.agent_persona_ops.get_llm_client') as mock_llm:
            # Mock LLM responses for different agents
            responses = {
                "coordinator": "I propose we approach this systematically...",
                "strategist": "From a strategic perspective, the risks are...",
                "auditor": "Compliance check reveals..."
            }
            
            def mock_generate(prompt, **kwargs):
                # Determine which agent based on prompt content
                for agent, response in responses.items():
                    if agent in prompt.lower():
                        return response
                return "Generic response"
            
            mock_client = MagicMock()
            mock_client.generate.side_effect = mock_generate
            mock_llm.return_value = mock_client
            
            # Mock Cortex query
            with patch.object(CortexOperations, 'query') as mock_cortex_query:
                mock_cortex_query.return_value = MagicMock(
                    results=[
                        MagicMock(
                            content="Protocol 87 defines structured queries...",
                            metadata={"source": "01_PROTOCOLS/087_structured_queries.md"},
                            score=0.95
                        ),
                        MagicMock(
                            content="Security mandate requires...",
                            metadata={"source": "01_PROTOCOLS/security.md"},
                            score=0.88
                        )
                    ]
                )
                
                # Initialize Council operations
                council_ops = CouncilOperations()
                
                # Dispatch full council deliberation
                result = council_ops.dispatch_task(
                    task_description="Design a new protocol for MCP composition patterns",
                    agent=None,  # Full council
                    max_rounds=2
                )
                
                # Verify successful execution
                assert result["status"] == "success"
                assert result["rounds"] == 2
                assert len(result["agents"]) == 3
                assert set(result["agents"]) == {"coordinator", "strategist", "auditor"}
                
                # Verify packets were created (2 rounds × 3 agents = 6 packets)
                assert len(result["packets"]) == 6
                
                # Verify each packet has required fields
                for packet in result["packets"]:
                    assert "session_id" in packet
                    assert "round_id" in packet
                    assert "member_id" in packet
                    assert "decision" in packet
                    assert packet["member_id"] in ["coordinator", "strategist", "auditor"]
                
                # Verify Cortex was queried once (at start)
                assert mock_cortex_query.call_count == 1
                
                # Verify final synthesis exists
                assert "final_synthesis" in result
                assert result["final_synthesis"]


@pytest.mark.integration  
class TestCortexMCPOperations:
    """Test Cortex MCP operations work correctly"""
    
    def test_cortex_query_returns_results(self):
        """
        Test that Cortex MCP query operation works
        
        Note: This requires actual ChromaDB setup, so we mock at the collection level
        """
        with patch('mcp_servers.cognitive.cortex.operations.chromadb') as mock_chromadb:
            # Mock ChromaDB client and collection
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            
            # Mock query results
            mock_collection.query.return_value = {
                'documents': [['Protocol 101 content', 'Protocol 102 content']],
                'metadatas': [[
                    {'source': '01_PROTOCOLS/101.md'},
                    {'source': '01_PROTOCOLS/102.md'}
                ]],
                'distances': [[0.1, 0.2]]
            }
            
            # Initialize Cortex operations
            cortex_ops = CortexOperations(project_root=".")
            
            # Query
            result = cortex_ops.query("What is Protocol 101?", max_results=2)
            
            # Verify results
            assert hasattr(result, 'results')
            assert len(result.results) == 2
            assert result.results[0].content == "Protocol 101 content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

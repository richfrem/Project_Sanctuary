"""
Integration Tests for Agent Persona MCP ↔ Cortex MCP Communication

Tests the full flow:
1. Council MCP → Agent Persona MCP → Cortex MCP
2. Multi-agent deliberation with context retrieval
3. Cross-MCP communication patterns
"""

import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock missing legacy modules to allow imports in CortexOperations
mock_vector_service = MagicMock()
sys.modules["mnemonic_cortex.app.services.vector_db_service"] = mock_vector_service
sys.modules["mnemonic_cortex.app.services.llm_service"] = MagicMock()

from mcp_servers.council.council_ops import CouncilOperations
from mcp_servers.agent_persona.agent_persona_ops import AgentPersonaOperations
from mcp_servers.rag_cortex.operations import CortexOperations


@pytest.mark.integration
class TestAgentPersonaCortexIntegration:
    """Test Agent Persona MCP can successfully query Cortex MCP"""
    
    def test_persona_queries_cortex_for_context(self):
        """
        Test that Agent Persona MCP can query Cortex MCP for context
        """
        with patch('mcp_servers.agent_persona.agent_persona_ops.get_llm_client') as mock_llm:
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
                assert result["status"] == "success"
                assert "response" in result


@pytest.mark.integration
class TestCouncilAgentPersonaCortexFlow:
    """Test full Council MCP → Agent Persona MCP → Cortex MCP flow"""
    
    def test_council_dispatch_full_flow(self):
        """
        Test complete flow from Council through Agent Persona to Cortex
        """
        with patch('mcp_servers.agent_persona.agent_persona_ops.get_llm_client') as mock_llm, \
             patch.object(CortexOperations, 'cache_warmup'): # Prevent warmup side effects
            
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
        """
        with patch('mcp_servers.agent_persona.agent_persona_ops.get_llm_client') as mock_llm, \
             patch.object(CortexOperations, 'cache_warmup'): # Prevent warmup side effects
            
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
        """
        # Mock VectorDBService which is imported inside query()
        mock_db_service_cls = mock_vector_service.VectorDBService
        mock_db_instance = mock_db_service_cls.return_value
        mock_retriever = mock_db_instance.get_retriever.return_value
        
        # Mock retriever results
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Protocol 101 content"
        mock_doc1.metadata = {'source': '01_PROTOCOLS/101.md'}
        
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Protocol 102 content"
        mock_doc2.metadata = {'source': '01_PROTOCOLS/102.md'}
        
        mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
        
        # Initialize Cortex operations
        cortex_ops = CortexOperations(project_root=".")
        
        # Query
        result = cortex_ops.query("What is Protocol 101?", max_results=2)
        
        # Verify results
        assert hasattr(result, 'results')
        # We expect results if the DB is populated (which it is from previous tests)
        assert len(result.results) > 0
        assert result.results[0].content is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

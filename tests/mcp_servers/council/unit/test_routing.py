"""
Test suite for Council MCP polymorphic model routing (T094)

Verifies Protocol 116 (Container Network Isolation) compliance
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mcp_servers.agent_persona.llm_client import OllamaClient, get_llm_client
from mcp_servers.agent_persona.agent_persona_ops import AgentPersonaOperations
from mcp_servers.council.council_ops import CouncilOperations


class TestOllamaClientProtocol116:
    """Test OllamaClient Protocol 116 compliance"""
    
    def test_default_uses_container_network(self):
        """Verify default Ollama host uses container network addressing"""
        with patch.dict('os.environ', {}, clear=True):
            client = OllamaClient()
            assert client.host == "http://ollama_model_mcp:11434"
    
    def test_explicit_ollama_host_parameter(self):
        """Verify explicit ollama_host parameter takes precedence"""
        client = OllamaClient(ollama_host="http://custom-host:11434")
        assert client.host == "http://custom-host:11434"
    
    def test_env_var_overrides_default(self):
        """Verify OLLAMA_HOST env var overrides default"""
        with patch.dict('os.environ', {'OLLAMA_HOST': 'http://env-host:11434'}):
            client = OllamaClient()
            assert client.host == "http://env-host:11434"
    
    @patch('mcp_servers.agent_persona.llm_client.logger')
    def test_localhost_warning(self, mock_logger):
        """Verify localhost usage triggers Protocol 116 warning"""
        client = OllamaClient(ollama_host="http://localhost:11434")
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Protocol 116" in warning_msg
        assert "ollama_model_mcp:11434" in warning_msg


class TestAgentPersonaModelPreference:
    """Test Agent Persona MCP model_preference routing"""
    
    @patch('mcp_servers.agent_persona.agent_persona_ops.get_llm_client')
    @patch('mcp_servers.agent_persona.agent_persona_ops.Agent')
    def test_ollama_preference_uses_container_network(self, mock_agent, mock_get_client):
        """Verify model_preference='OLLAMA' routes to container network"""
        ops = AgentPersonaOperations()
        
        # Mock agent query
        mock_agent_instance = Mock()
        mock_agent_instance.query.return_value = "Test response"
        mock_agent.return_value = mock_agent_instance
        
        # Dispatch with OLLAMA preference
        ops.dispatch(
            role="coordinator",
            task="Test task",
            model_preference="OLLAMA"
        )
        
        # Verify get_llm_client was called with ollama_host
        mock_get_client.assert_called_once()
        call_kwargs = mock_get_client.call_args[1]
        assert call_kwargs['ollama_host'] == "http://ollama_model_mcp:11434"
    
    @patch('mcp_servers.agent_persona.agent_persona_ops.get_llm_client')
    @patch('mcp_servers.agent_persona.agent_persona_ops.Agent')
    def test_no_preference_no_ollama_host(self, mock_agent, mock_get_client):
        """Verify no model_preference doesn't set ollama_host"""
        ops = AgentPersonaOperations()
        
        mock_agent_instance = Mock()
        mock_agent_instance.query.return_value = "Test response"
        mock_agent.return_value = mock_agent_instance
        
        ops.dispatch(
            role="coordinator",
            task="Test task"
        )
        
        # Verify ollama_host is None
        call_kwargs = mock_get_client.call_args[1]
        assert call_kwargs['ollama_host'] is None


class TestCouncilModelPreference:
    """Test Council MCP model_preference parameter threading"""
    
    @patch('mcp_servers.agent_persona.agent_persona_ops.AgentPersonaOperations')
    @patch('mcp_servers.rag_cortex.operations.CortexOperations')
    def test_model_preference_passed_to_persona_ops(self, mock_cortex, mock_persona):
        """Verify model_preference is passed through to Agent Persona MCP"""
        # Setup mocks
        mock_persona_instance = Mock()
        mock_persona_instance.dispatch.return_value = {
            "response": "Test response",
            "status": "success"
        }
        mock_persona_instance.list_roles.return_value = {
            "built_in": ["coordinator", "strategist", "auditor"],
            "custom": []
        }
        mock_persona.return_value = mock_persona_instance
        
        mock_cortex_instance = Mock()
        mock_cortex_instance.query.return_value = {"results": []}
        mock_cortex_instance.get_cache_stats.return_value = {"hot_cache_size": 10}
        mock_cortex.return_value = mock_cortex_instance
        
        # Create Council ops and dispatch
        ops = CouncilOperations()
        ops.dispatch_task(
            task_description="Test task",
            agent="coordinator",
            model_preference="OLLAMA"
        )
        
        # Verify persona_ops.dispatch was called with model_preference
        mock_persona_instance.dispatch.assert_called()
        call_kwargs = mock_persona_instance.dispatch.call_args[1]
        assert call_kwargs['model_preference'] == "OLLAMA"


class TestFactoryFunction:
    """Test get_llm_client factory function"""
    
    def test_factory_passes_ollama_host(self):
        """Verify factory function passes ollama_host to OllamaClient"""
        client = get_llm_client(
            provider="ollama",
            ollama_host="http://test-host:11434"
        )
        assert isinstance(client, OllamaClient)
        assert client.host == "http://test-host:11434"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

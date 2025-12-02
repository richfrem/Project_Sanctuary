import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.lib.council.council_ops import CouncilOperations

@pytest.mark.integration
def test_council_git_flow():
    """Test Council directing Git operations."""
    # Mock Git MCP client
    with patch('mcp_servers.lib.council.council_ops.get_mcp_client') as mock_get_client:
        # Setup mock client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Initialize operations
        council = CouncilOperations()
        
        # Simulate task that requires git commit
        # We mock the LLM response to ensure deterministic behavior
        with patch.object(council, '_query_llm') as mock_llm:
            mock_llm.return_value = {
                "decision": "I will create a feature branch.",
                "tool_calls": [
                    {
                        "name": "git_create_branch",
                        "arguments": {"branch_name": "feat/new-protocol"}
                    }
                ]
            }
            
            # Dispatch task
            result = council.dispatch_task(
                "Create a feature branch for new protocol",
                agent="coordinator",
                max_rounds=1
            )
            
            # Verify Git MCP tool was called via the client
            # Note: The actual implementation might differ slightly depending on how Council calls tools
            # This assumes Council uses a standard client wrapper
            # If Council uses direct tool calls, we might need to adjust the mock
            
            # For now, we assume the Council orchestrator identifies the tool call and executes it
            # or passes it back. If it executes it, it would use the client.
            
            # Since we are mocking the LLM to return a tool call, the Council logic should
            # attempt to execute that tool call using the appropriate MCP client.
            
            # Check if get_mcp_client was called for 'git'
            mock_get_client.assert_called_with("git")
            
            # Check if the tool was called on the client
            mock_client.call_tool.assert_called_with(
                "git_create_branch", 
                {"branch_name": "feat/new-protocol"}
            )

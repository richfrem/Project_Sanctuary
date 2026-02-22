"""
Unit tests for Sanctuary Network Cluster (Business Logic).
Mocks httpx to avoid external network calls.
"""
import pytest
from unittest.mock import AsyncMock, patch
from mcp_servers.gateway.clusters.sanctuary_network import tools

class TestSanctuaryNetwork:
    @pytest.mark.asyncio
    async def test_fetch_url(self):
        # Mock httpx.AsyncClient context manager
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            
            # Mock get response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "Success Content"
            mock_client.get.return_value = mock_response

            result = await tools.fetch_url("http://example.com")
            
            assert "Status: 200" in result
            assert "Content:\nSuccess Content" in result
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_site_status(self):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.head.return_value = mock_response

            result = await tools.check_site_status("http://example.com")
            
            assert "is UP" in result
            assert "Status: 200" in result
            mock_client.head.assert_called_once()
            
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

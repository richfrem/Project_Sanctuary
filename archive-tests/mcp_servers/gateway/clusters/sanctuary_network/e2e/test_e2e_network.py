"""
E2E Tests for sanctuary_network cluster (2 tools)

Tools tested:
- fetch-url
- check-site-status
"""
import pytest


# =============================================================================
# NETWORK TOOLS (2)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestNetworkTools:
    
    def test_fetch_url(self, logged_call):
        """Test fetch-url retrieves content from a URL."""
        result = logged_call("sanctuary-network-fetch-url", {
            "url": "https://httpbin.org/get"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
        content = str(result["result"].get("content", []))
        # httpbin.org/get returns JSON with request details
        assert "origin" in content.lower() or "url" in content.lower(), f"Expected httpbin response"
    
    def test_check_site_status(self, logged_call):
        """Test check-site-status verifies site availability."""
        result = logged_call("sanctuary-network-check-site-status", {
            "url": "https://httpbin.org/status/200"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"

import pytest
import requests
import os
import urllib3
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest

# Load environment variables from .env file
load_dotenv()

# Suppress InsecureRequestWarning for self-signed certs (Localhost/Podman)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@pytest.mark.integration
@pytest.mark.gateway
class TestGatewayBlackBox(BaseIntegrationTest):
    """
    "Black Box" verification suite for the decoupled Sanctuary Gateway.
    Inherits from BaseIntegrationTest for standard dependency checking.
    """
    
    def get_required_services(self) -> List[Tuple[str, int, str]]:
        """Declare dependency on the external Gateway container."""
        return [("localhost", 4444, "Sanctuary Gateway")]

    @property
    def config(self):
        """Lazy load config."""
        return {
            "URL": os.getenv("MCP_GATEWAY_URL", "https://localhost:4444"),
            "API_TOKEN": os.getenv("MCPGATEWAY_BEARER_TOKEN", "")
        }

    def test_pulse_check(self):
        """
        1. The 'Pulse' Check
        Target: GET /health
        Expectation: 200 OK
        """
        url = f"{self.config['URL']}/health"
        print(f"\n[Pulse] Checking heartbeat at: {url}")

        try:
            response = requests.get(url, verify=False, timeout=5)
            assert response.status_code == 200, \
                f"Pulse Check Failed: Expected 200 OK, got {response.status_code}. Body: {response.text}"
        except requests.exceptions.ConnectionError:
            pytest.fail(f"Pulse Check Failed: Connection refused to {url}")

    def test_circuit_breaker(self):
        """
        2. The 'Circuit Breaker' Check
        Target: GET /tools
        Condition: Invalid API Token
        Expectation: 401 Unauthorized or 403 Forbidden
        """
        url = f"{self.config['URL']}/tools"
        headers = {"Authorization": "Bearer invalid-token-should-be-rejected"}
        
        print(f"\n[Circuit Breaker] Testing security with invalid token at: {url}")
        response = requests.get(url, headers=headers, verify=False, timeout=5)

        assert response.status_code in [401, 403], \
            f"Circuit Breaker Failed! Gateway accepted invalid token. Status: {response.status_code}"

    def test_handshake(self):
        """
        3. The 'Handshake' Check
        Target: GET /tools
        Condition: Valid API Token
        Expectation: 200 OK
        """
        if not self.config['API_TOKEN']:
            pytest.skip("MCPGATEWAY_BEARER_TOKEN not set in environment")
        
        url = f"{self.config['URL']}/tools"
        headers = {"Authorization": f"Bearer {self.config['API_TOKEN']}"}
        
        print(f"\n[Handshake] Authenticating with API token at: {url}")

        try:
            response = requests.get(url, headers=headers, verify=False, timeout=5)
            assert response.status_code == 200, \
                f"Handshake Failed: Token rejected. Status: {response.status_code}. Body: {response.text}"
        except requests.exceptions.ConnectionError:
            pytest.fail(f"Handshake Failed: Connection refused at {url}")

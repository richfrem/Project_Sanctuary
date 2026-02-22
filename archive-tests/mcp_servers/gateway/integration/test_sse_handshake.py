"""
SSE Handshake Verification Tests (Tier 2 - Integration)

Purpose: Validates all Gateway cluster SSE endpoints respond with proper
MCP-compliant handshake (event: endpoint + data: /messages).

Reference: ADR-066 v1.3, Task 146 Phase 5
"""
import pytest
import httpx
from typing import Tuple, List

# Fleet configuration: (name, external_port, internal_port)
GATEWAY_FLEET: List[Tuple[str, int, int]] = [
    ("sanctuary_utils", 8100, 8000),
    ("sanctuary_filesystem", 8101, 8000),
    ("sanctuary_network", 8102, 8000),
    ("sanctuary_git", 8103, 8000),
    ("sanctuary_cortex", 8104, 8000),
    ("sanctuary_domain", 8105, 8105),
]


class TestSSEHandshake:
    """Tier 2: Validates SSE server readiness for each Gateway cluster."""
    
    @pytest.fixture
    def timeout(self) -> float:
        return 5.0
    
    @pytest.mark.parametrize("server_name,port,_", GATEWAY_FLEET)
    def test_health_endpoint(self, server_name: str, port: int, _: int):
        """Each server must have a /health endpoint returning 200 OK."""
        url = f"http://localhost:{port}/health"
        try:
            response = httpx.get(url, timeout=5.0)
            assert response.status_code == 200, f"{server_name}: /health returned {response.status_code}"
            data = response.json()
            assert data.get("status") in ["ok", "healthy"], f"{server_name}: unexpected health status: {data}"
        except httpx.ConnectError:
            pytest.skip(f"{server_name} not running on port {port}")
    
    @pytest.mark.parametrize("server_name,port,_", GATEWAY_FLEET)
    def test_sse_handshake(self, server_name: str, port: int, _: int):
        """
        Verify SSE endpoint returns proper MCP handshake.
        
        Per ADR-066 v1.3:
        1. Client connects to /sse (GET, persistent connection)
        2. Server IMMEDIATELY sends 'endpoint' event with POST URL
        3. Connection stays open with periodic heartbeat pings
        
        Expected stream:
            event: endpoint
            data: /messages
        """
        url = f"http://localhost:{port}/sse"
        try:
            with httpx.stream("GET", url, timeout=5.0) as response:
                assert response.status_code == 200, f"{server_name}: SSE returned {response.status_code}"
                
                content_type = response.headers.get("content-type", "")
                assert "text/event-stream" in content_type, \
                    f"{server_name}: Expected text/event-stream, got {content_type}"
                
                # Read first two lines
                lines = []
                for line in response.iter_lines():
                    lines.append(line)
                    if len(lines) >= 2:
                        break
                
                assert len(lines) >= 2, f"{server_name}: SSE stream too short: {lines}"
                assert lines[0] == "event: endpoint", \
                    f"{server_name}: Expected 'event: endpoint', got '{lines[0]}'"
                assert lines[1] == "data: /messages", \
                    f"{server_name}: Expected 'data: /messages', got '{lines[1]}'"
                    
        except httpx.ConnectError:
            pytest.skip(f"{server_name} not running on port {port}")


class TestFleetHealth:
    """Aggregate fleet health verification."""
    
    def test_all_servers_healthy(self):
        """Verify all 6 Gateway clusters are healthy."""
        healthy = []
        unhealthy = []
        
        for name, port, _ in GATEWAY_FLEET:
            try:
                response = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
                if response.status_code == 200:
                    healthy.append(name)
                else:
                    unhealthy.append((name, f"HTTP {response.status_code}"))
            except httpx.ConnectError:
                unhealthy.append((name, "Connection refused"))
            except httpx.ReadTimeout:
                unhealthy.append((name, "Timeout"))
        
        assert len(unhealthy) == 0, f"Unhealthy servers: {unhealthy}"
        assert len(healthy) == 6, f"Expected 6 healthy servers, got {len(healthy)}: {healthy}"
    
    def test_all_sse_endpoints_streaming(self):
        """Verify all 6 Gateway clusters have working SSE endpoints."""
        working = []
        broken = []
        
        for name, port, _ in GATEWAY_FLEET:
            try:
                with httpx.stream("GET", f"http://localhost:{port}/sse", timeout=3.0) as resp:
                    first_line = next(resp.iter_lines(), None)
                    if first_line == "event: endpoint":
                        working.append(name)
                    else:
                        broken.append((name, f"Unexpected: {first_line}"))
            except httpx.ConnectError:
                broken.append((name, "Connection refused"))
            except StopIteration:
                broken.append((name, "Empty stream"))
            except Exception as e:
                broken.append((name, str(e)))
        
        assert len(broken) == 0, f"Broken SSE endpoints: {broken}"
        assert len(working) == 6, f"Expected 6 working SSE endpoints, got {len(working)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

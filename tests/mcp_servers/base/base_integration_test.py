import pytest
import socket
import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseIntegrationTest(ABC):
    """
    Base class for Live Integration Tests (ADR 047/048).
    
    These tests sit at Layer 2 of the Testing Pyramid.
    They validate REAL connectivity to external dependencies (ChromaDB, Ollama, Git)
    before the MCP API layer is tested.
    
    Usage:
        class TestRAGCortexLive(LiveIntegrationTest):
            def get_required_services(self):
                return [("localhost", 8110, "ChromaDB")]
                
            def test_ingest(self):
                ...
    """
    
    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """
        Automatically check dependencies before running any test in this class.
        Skips the test if dependencies are missing (unless CI=true).
        """
        services = self.get_required_services()
        missing = []
        
        for host, port, name in services:
            if not self._is_port_open(host, port):
                missing.append(f"{name} ({host}:{port})")
                
        if missing:
            msg = f"Required services not running: {', '.join(missing)}"
            if os.environ.get("CI") == "true":
                pytest.fail(msg)
            else:
                pytest.skip(msg)

    @abstractmethod
    def get_required_services(self) -> List[Tuple[str, int, str]]:
        """
        Return a list of (host, port, service_name) tuples required for these tests.
        Example: [("localhost", 8110, "ChromaDB")]
        """
        pass

    def _is_port_open(self, host: str, port: int, timeout: int = 1) -> bool:
        """Check if a TCP port is open."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

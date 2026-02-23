
import logging
from pathlib import Path
from typing import Dict, Optional
import os
from tests.mcp_servers.base.mcp_test_client import MCPTestClient
from mcp_servers.start_mcp_servers import MODULES_TO_START, PROJECT_ROOT

logger = logging.getLogger(__name__)

class MCPServerFleet:
    """
    Manages a fleet of MCP servers for E2E testing.
    Spawns them all as subprocesses and provides access to their clients.
    """
    def __init__(self):
        self.clients: Dict[str, MCPTestClient] = {}
        
    def start_all(self, modules: Optional[list] = None):
        """Start known MCP servers (all or specified subset)."""
        logger.info("Starting MCP Server Fleet...")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        
        # Add RAG Cortex environment variables (defaults for testing)
        env["CHROMA_HOST"] = env.get("CHROMA_HOST", "localhost")
        env["CHROMA_PORT"] = env.get("CHROMA_PORT", "8000")
        env["CHROMA_CHILD_COLLECTION"] = env.get("CHROMA_CHILD_COLLECTION", "child_chunks_v5")
        env["CHROMA_PARENT_STORE"] = env.get("CHROMA_PARENT_STORE", "parent_documents_v5")
        env["SKIP_CONTAINER_CHECKS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        
        modules_to_start = modules if modules else MODULES_TO_START
        
        for mod_name in modules_to_start:
            # mod_name is like "mcp_servers.chronicle.server"
            # We want short name "chronicle"
            short_name = mod_name.split(".")[1] 
            
            # Start as module to support relative imports
            logger.info(f"Booting {short_name}...")
            client = MCPTestClient(mod_name, is_module=True)
            client.start(env=env)
            self.clients[short_name] = client
        
        # Warmup phase: Pre-load heavy dependencies
        self._warmup_servers()
            
        logger.info(f"Fleet started with {len(self.clients)} servers.")
    
    def _warmup_servers(self):
        """Pre-warm servers to trigger lazy loading of heavy dependencies."""
        logger.info("Warming up servers...")
        
        # RAG Cortex: Trigger CortexOperations initialization
        if "rag_cortex" in self.clients:
            try:
                logger.info("Pre-loading RAG Cortex dependencies (ChromaDB, LangChain, embeddings)...")
                import time
                start = time.time()
                
                # Call a lightweight operation that triggers full initialization
                self.clients["rag_cortex"].call_tool("cortex_get_stats", {})
                
                elapsed = time.time() - start
                logger.info(f"RAG Cortex warmed up in {elapsed:.2f}s")
            except Exception as e:
                logger.warning(f"RAG Cortex warmup failed (non-fatal): {e}")
        
        logger.info("Server warmup complete.")
    
    def stop_all(self):
        """Stop all running servers."""
        logger.info("Stopping MCP Server Fleet...")
        for name, client in self.clients.items():
            logger.info(f"Stopping {name}...")
            try:
                client.stop()
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        self.clients.clear()

    def get_client(self, server_name: str) -> MCPTestClient:
        """Get a specific server client by name (e.g. 'code', 'chronicle')."""
        if server_name not in self.clients:
            raise KeyError(f"Server '{server_name}' not running in fleet. Available: {list(self.clients.keys())}")
        return self.clients[server_name]
    
    def call_tool(self, server_name: str, tool_name: str, args: dict = {}) -> any:
        """Helper to call a tool on a specific server."""
        client = self.get_client(server_name)
        return client.call_tool(tool_name, args)

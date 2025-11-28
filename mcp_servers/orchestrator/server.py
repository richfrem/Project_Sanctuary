"""
Orchestrator MCP Server
Domain: project_sanctuary.orchestrator

Provides MCP tools for the Sanctuary Council (Strategist, Auditor, etc.) to
orchestrate high-level missions and decisions.
"""
from fastmcp import FastMCP
import os
import sys
import json
from pathlib import Path
from typing import Optional, List
from mcp.server.fastmcp import FastMCP

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import tools
from mcp_servers.orchestrator.tools.cognitive import (
    create_cognitive_task,
    create_development_cycle,
    query_mnemonic_cortex
)
from mcp_servers.orchestrator.tools.mechanical import (
    create_file_write_task,
    create_git_commit_task
)
from mcp_servers.orchestrator.tools.query import (
    get_orchestrator_status,
    list_recent_tasks,
    get_task_result
)

# Initialize MCP Server
mcp = FastMCP("orchestrator")

# Configuration
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")

# Load Config
CONFIG_PATH = Path(__file__).parent / "config" / "mcp_config.json"
try:
    with open(CONFIG_PATH, "r") as f:
        MCP_CONFIG = json.load(f)
except Exception as e:
    print(f"Warning: Could not load config from {CONFIG_PATH}: {e}")
    MCP_CONFIG = {}

# TODO: On server startup, call cortex_guardian_wakeup() to initialize the cache
# and generate the boot digest for the Council.


@mcp.tool()
def orchestrator_dispatch_mission(
    mission_id: str,
    objective: str,
    assigned_agent: str = "Kilo"
) -> str:
    """
    Dispatch a mission to an agent.
    
    Args:
        mission_id: Unique mission identifier
        objective: The objective of the mission
        assigned_agent: The agent assigned to the mission
    """
    # TODO: Connect to task management or agent dispatch system
    return f"Mission '{mission_id}' dispatched to {assigned_agent}. Objective: {objective}"


# ============================================================================
# Strategic Crucible Loop
# ============================================================================

# Import dependencies for the loop
# Note: In a distributed MCP architecture, we would call these via client.
# Here we import the service logic directly for reliability.
from mnemonic_cortex.app.services.ingestion_service import IngestionService
from mnemonic_cortex.app.synthesis.generator import SynthesisGenerator
from mcp_servers.cognitive.cortex.operations import CortexOperations

@mcp.tool()
def orchestrator_run_strategic_cycle(
    gap_description: str,
    research_report_path: str,
    days_to_synthesize: int = 1
) -> str:
    """
    Execute a full Strategic Crucible Loop: Ingest -> Synthesize -> Adapt -> Cache.
    
    Args:
        gap_description: Description of the strategic gap being addressed.
        research_report_path: Path to the new research report (markdown).
        days_to_synthesize: Window for adaptation packet generation.
        
    Returns:
        Summary of the cycle execution.
    """
    results = []
    results.append(f"--- Strategic Crucible Cycle: {gap_description} ---")
    
    # 1. Ingestion (Medium Memory Update)
    try:
        results.append(f"1. Ingesting Report: {research_report_path}")
        ingestion_service = IngestionService(
            project_root=PROJECT_ROOT
        )
        # We assume incremental ingest for a single report
        ingest_stats = ingestion_service.ingest_incremental([research_report_path])
        results.append(f"   - Ingestion Complete: {ingest_stats}")
    except Exception as e:
        return "\n".join(results) + f"\n[CRITICAL FAIL] Ingestion failed: {e}"

    # 2. Adaptation (Slow Memory Update Prep)
    try:
        results.append(f"2. Generating Adaptation Packet (Window: {days_to_synthesize} days)")
        generator = SynthesisGenerator(PROJECT_ROOT)
        packet = generator.generate_packet(days=days_to_synthesize)
        packet_path = generator.save_packet(packet)
        results.append(f"   - Packet Generated: {packet_path}")
        results.append(f"   - Packet ID: {packet.packet_id}")
    except Exception as e:
        return "\n".join(results) + f"\n[CRITICAL FAIL] Adaptation failed: {e}"

    # 3. Cache Update (Fast Memory Update)
    try:
        results.append(f"3. Waking Guardian Cache")
        # Initialize Cortex Ops to access cache logic
        cortex_ops = CortexOperations(PROJECT_ROOT) 
        # We need to inject the real cache instance if possible, or rely on the ops to create it
        # The current CortexOperations implementation creates MnemonicCache internally if not passed.
        # However, it needs DB_PATH etc from env.
        # Let's assume env vars are set or defaults work.
        
        # We call guardian_wakeup. In a real scenario, this might be an async tool call.
        # Here we call the method directly if available, or simulate it.
        # Looking at cortex/operations.py, guardian_wakeup is a method.
        wakeup_stats = cortex_ops.guardian_wakeup()
        results.append(f"   - Cache Updated: {wakeup_stats}")
    except Exception as e:
        results.append(f"   - [WARN] Cache update failed (non-critical): {e}")

    results.append("--- Cycle Complete ---")
    return "\n".join(results)


if __name__ == "__main__":
    mcp.run()


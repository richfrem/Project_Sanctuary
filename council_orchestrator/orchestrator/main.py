# council_orchestrator/orchestrator/main.py
# Main entry point for the council orchestrator

import asyncio
import sys
from .app import Orchestrator

def main():
    """Main entry point for the council orchestrator."""
    # Initialize orchestrator
    orchestrator = Orchestrator()

    try:
        # Main execution loop
        asyncio.run(orchestrator.main_loop())
    except KeyboardInterrupt:
        orchestrator.logger.info("Orchestrator shutdown via keyboard interrupt")
    except Exception as e:
        orchestrator.logger.error(f"Critical orchestrator failure: {e}")
        raise

if __name__ == "__main__":
    main()
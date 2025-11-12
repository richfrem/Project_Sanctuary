# council_orchestrator/orchestrator/main.py
# Main entry point for the council orchestrator

import asyncio
import sys
import argparse
from .app import Orchestrator

def main():
    """Main entry point for the council orchestrator."""
    # --- NEW: Add argument parser for --one-shot flag ---
    parser = argparse.ArgumentParser(description="Sanctuary Council Orchestrator")
    parser.add_argument(
        '--one-shot',
        action='store_true',
        help='Run the orchestrator for a single command and then exit.'
    )
    args = parser.parse_args()
    # --- END NEW ---

    # Initialize orchestrator, passing the one_shot flag
    orchestrator = Orchestrator(one_shot=args.one_shot)

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
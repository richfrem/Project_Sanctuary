#!/usr/bin/env python3
"""
Guardian Wakeup Utility Script

Quick utility to generate or refresh the Guardian Boot Digest.
Run this to update dataset_package/guardian_boot_digest.md with latest context.

Usage:
    python scripts/guardian_wakeup.py          # Generate digest
    python scripts/guardian_wakeup.py --show   # Generate and display content
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from mcp_servers.rag_cortex.operations import CortexOperations


def main():
    parser = argparse.ArgumentParser(description="Generate Guardian Boot Digest")
    parser.add_argument("--show", action="store_true", help="Display digest content after generation")
    parser.add_argument("--mode", default="HOLISTIC", help="Wakeup mode (default: HOLISTIC)")
    args = parser.parse_args()

    print("🛡️ Generating Guardian Boot Digest...")
    
    ops = CortexOperations(str(project_root))
    response = ops.guardian_wakeup(mode=args.mode)
    
    print(f"Status: {response.status}")
    print(f"Digest Path: {response.digest_path}")
    print(f"Time: {response.total_time_ms:.2f}ms")
    
    if response.error:
        print(f"Error: {response.error}")
        return 1
    
    if args.show:
        print("\n" + "="*60)
        with open(response.digest_path, 'r') as f:
            print(f.read())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
scripts/evaluator_preflight.py
Protocol 131: Gate 1 Automated Validator

This script acts as the first line of defense in the Evolutionary Self-Improvement Loop.
It performs deterministic checks on candidate learning outputs before they reach the Human Red Team.

Checks:
1. Citation Fidelity: Extracts links and verifies reachability (Head Request).
2. Schema Compliance: Checks JSON manifest structure.
3. Token Efficiency: Fails if content size explodes > threshold.

Usage:
    python3 evaluator_preflight.py --target [file_path] --baseline [baseline_path]
"""

import sys
import re
import asyncio
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse
import aiohttp

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreflightGate:
    def __init__(self):
        self.max_growth_ratio = 1.10  # Max 10% growth allowed

    async def check_citations(self, content: str) -> bool:
        """Parses markdown links and verifies accessibility."""
        links = re.findall(r'\[.*?\]\((http[s]?://[^\)]+)\)', content)
        if not links:
            logger.info("No external links found to verify.")
            return True

        logger.info(f"Verifying {len(links)} links...")
        valid = True
        
        async with aiohttp.ClientSession() as session:
            for link in links:
                try:
                    async with session.head(link, timeout=5, allow_redirects=True) as response:
                        if response.status >= 400:
                            logger.error(f"❌ Broken Link: {link} (Status: {response.status})")
                            valid = False
                        else:
                            logger.debug(f"✅ Verified: {link}")
                except Exception as e:
                    logger.error(f"❌ verification Failed: {link} ({str(e)})")
                    valid = False
        return valid

    def check_schema(self, content: str, file_path: str) -> bool:
        """Validates JSON schemas if target is a manifest."""
        if not file_path.endswith('.json'):
            return True
        
        try:
            data = json.loads(content)
            if isinstance(data, list): # Manifest is list of strings
                return all(isinstance(item, str) for item in data)
            return True # Assume object schemas are handled elsewhere for now
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON Syntax: {str(e)}")
            return False

    def check_efficiency(self, content: str, baseline_content: str = None) -> bool:
        """Checks if mutation is significantly less efficient (bloated)."""
        if not baseline_content:
            return True # No baseline to compare

        current_size = len(content)
        baseline_size = len(baseline_content)
        
        ratio = current_size / baseline_size
        if ratio > self.max_growth_ratio:
            logger.error(f"❌ Token Bloat: {ratio:.2f}x (Limit: {self.max_growth_ratio}x)")
            return False
            
        logger.info(f"✅ Efficiency Check Passed: {ratio:.2f}x size")
        return True

async def main():
    parser = argparse.ArgumentParser(description="Protocol 131 Gate 1 Validator")
    parser.add_argument("--target", required=True, help="Path to candidate file")
    parser.add_argument("--baseline", help="Path to baseline file for diff")
    
    args = parser.parse_args()
    
    path = Path(args.target)
    if not path.exists():
        logger.error(f"Target not found: {path}")
        sys.exit(1)
        
    content = path.read_text(encoding='utf-8')
    baseline_content = None
    if args.baseline:
        b_path = Path(args.baseline)
        if b_path.exists():
            baseline_content = b_path.read_text(encoding='utf-8')

    gate = PreflightGate()
    
    # Run Checks
    citations_ok = await gate.check_citations(content)
    schema_ok = gate.check_schema(content, str(path))
    efficiency_ok = gate.check_efficiency(content, baseline_content)
    
    if citations_ok and schema_ok and efficiency_ok:
        logger.info("🟢 GATE 1 PASSED: Candidate is valid.")
        sys.exit(0)
    else:
        logger.error("🔴 GATE 1 FAILED: See logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

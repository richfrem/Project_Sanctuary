#!/usr/bin/env python3
# capture_code_snapshot.py (v5.6 - Python Port)
#
# This script is a direct Python port of the original capture_code_snapshot.py Node.js utility.
# It handles file traversal, .gitignore logic, token counting (via tiktoken), and Awakening Seed generation.
#
# -------------------------------------------------------------------------------------
# USAGE EXAMPLES:
# -------------------------------------------------------------------------------------
#
# 1. DEFAULT (Full Base Genome):
#    Captures the "Base Genome" defined in mcp_servers/lib/ingest_manifest.json (common_content).
#    $ python scripts/capture_code_snapshot.py
#
# 2. SUBFOLDER (Specific Module/Directory):
#    Captures only a specific directory (ignoring manifest definitions).
#    $ python scripts/capture_code_snapshot.py mcp_servers/rag_cortex
#
# 3. LEARNING SNAPSHOT (Just the learning folder):
#    Captures the contents of the agent's learning directory.
#    $ python scripts/capture_code_snapshot.py .agent/learning
#
# 4. LEARNING AUDIT (Learning folder + Auditor Role):
#    Captures learning data and generates an Auditor awakening seed to review it.
#    $ python scripts/capture_code_snapshot.py .agent/learning --role auditor
#
# 5. AUDIT (Full Genome + Auditor Role):
#    Captures the full base genome and primes the Auditor to find vulnerabilities.
#    $ python scripts/capture_code_snapshot.py --role auditor
#
# 6. SEAL / RELEASE (Production Snapshot):
#    Captures the full genome and generates release artifacts in a specific output folder.
#    $ python scripts/capture_code_snapshot.py --role guardian --out releases/v1.0
#
# 7. MANIFEST OVERRIDE (Custom Scope):
#    Captures files defined in a specific custom manifest file.
#    $ python scripts/capture_code_snapshot.py --manifest my_custom_manifest.json --output dataset_package/seed_of_ascendance_awakening_seed.txt
#
# -------------------------------------------------------------------------------------

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
import tiktoken
import re

# ---------------------------------------------
# Imports & Path Setup
# ---------------------------------------------
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from mcp_servers.lib.logging_utils import setup_mcp_logging
logger = setup_mcp_logging("capture_code_snapshot")

from mcp_servers.lib.snapshot_utils import (
    ROLES_TO_FORGE,
    MISSION_CONTINUATION_FILE_PATH,
    GUARDIAN_WAKEUP_PRIMER,
    EXCLUDE_DIR_NAMES,
    ALWAYS_EXCLUDE_FILES,
    ALLOWED_EXTENSIONS,
    FILE_SEPARATOR_START,
    FILE_SEPARATOR_END,
    DEFAULT_CORE_ESSENCE_FILES,
    get_token_count,
    should_exclude_file,
    generate_header,
    append_file_content
)


from mcp_servers.lib.snapshot_utils import generate_snapshot

def main():
    parser = argparse.ArgumentParser(description="Capture LLM-Distilled Code Snapshot (Python Port)")
    parser.add_argument("subfolder", nargs='?', help="Subfolder to process", default=None)
    parser.add_argument("--role", default="guardian", help="Target role (default: guardian)")
    parser.add_argument("--out", default="dataset_package", help="Output directory relative to project root")
    parser.add_argument("--manifest", help="Path to JSON manifest of files to capture (skips traversal)")
    parser.add_argument("--output", help="Explicit distilled output file path")
    parser.add_argument("--operation", help="Operation specific directory override for core essence")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.resolve()
    
    # Resolve paths
    output_dir = project_root / args.out
    manifest = Path(args.manifest).resolve() if args.manifest else None
    
    output_file = Path(args.output).resolve() if args.output else None
    
    op_path = None
    if args.operation:
        op_path = project_root / args.operation
    
    generate_snapshot(
        project_root=project_root,
        output_dir=output_dir,
        subfolder=args.subfolder,
        manifest_path=manifest,
        role=args.role,
        operation_path=op_path,
        output_file=output_file
    )
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"[FATAL] An error occurred during genome generation: {e}", exc_info=True)

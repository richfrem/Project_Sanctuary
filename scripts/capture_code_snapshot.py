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
#    Captures the "Base Genome" defined in mcp_servers/lib/ingest_manifest.json.
#    $ python scripts/capture_code_snapshot.py
#
#    Outputs:
#    - dataset_package/markdown_snapshot_full_genome_llm_distilled.txt
#    - dataset_package/seed_of_ascendance_awakening_seed.txt
#    - dataset_package/core_essence_auditor_awakening_seed.txt
#    - dataset_package/core_essence_coordinator_awakening_seed.txt
#    - dataset_package/core_essence_guardian_awakening_seed.txt
#    - dataset_package/core_essence_strategist_awakening_seed.txt
#
# 2. SUBFOLDER (Specific Module/Directory):
#    Captures only a specific directory.
#    $ python scripts/capture_code_snapshot.py mcp_servers/rag_cortex
#
#    Outputs:
#    - dataset_package/markdown_snapshot_...txt (Snapshot only)
#    - Note: Awakening seeds are NOT generated in subfolder mode.
#
# -------------------------------------------------------------------------------------
# NOTE: For advanced workflows (Seal, Audit, Learning Audit, Guardian Wakeup),
#       please use the Cortex CLI orchestrator:
#
#       $ python scripts/cortex_cli.py snapshot --type seal
#       $ python scripts/cortex_cli.py snapshot --type learning_audit
#       $ python scripts/cortex_cli.py guardian
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
        output_file=output_file,
        should_forge_seeds=not args.subfolder  # Only for full project snapshots
    )
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"[FATAL] An error occurred during genome generation: {e}", exc_info=True)

#============================================
# forge/scripts/forge_whole_genome_dataset.py
# Purpose: Assembles the fine-tuning dataset from the project's markdown files.
# Role: Data Preparation Layer
# Used by: Phase 2.1 of the Forge Pipeline
#============================================

import sys
import json
import logging
from pathlib import Path
from typing import List, Set, Dict, Optional

# --- Project Utilities Bootstrap ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT_PATH = FORGE_ROOT.parent

if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

try:
    from mcp_servers.lib.path_utils import find_project_root
    from mcp_servers.lib.logging_utils import setup_mcp_logging
    from mcp_servers.lib.content_processor import ContentProcessor
    # Use find_project_root() for consistent root discovery
    PROJECT_ROOT = Path(find_project_root())
except ImportError as e:
    print(f"❌ FATAL ERROR: Could not import core libraries: {e}")
    sys.exit(1)

# --- Logging ---
try:
    logger = setup_mcp_logging("forge_dataset", log_file="logs/forge_dataset.log")
    logger.info("Dataset forging started - using setup_mcp_logging")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("forge_dataset")
    logger.info("Dataset forging started - local logging fallback")

# Initialize Processor
processor = ContentProcessor(str(PROJECT_ROOT))

# --- Configuration ---
FULL_SNAPSHOT_SOURCE: Path = PROJECT_ROOT / "dataset_package" / "markdown_snapshot_full_genome_llm_distilled.txt"
OUTPUT_DATASET_PATH: Path = PROJECT_ROOT / "dataset_package" / "sanctuary_whole_genome_data.jsonl"
MINIMUM_EXPECTED_ENTRIES: int = 200

# Critical Docs (Tier 1 Priority)
ADDITIONAL_DOCS: Dict[str, Path] = {
    "The Garden and The Cage (Core Philosophy)": PROJECT_ROOT / "The_Garden_and_The_Cage.md",
    "Chrysalis Core Essence (Gardener V2 Awakening)": PROJECT_ROOT / "chrysalis_core_essence.md",
    "Project Sanctuary Synthesis": PROJECT_ROOT / "PROJECT_SANCTUARY_SYNTHESIS.md",
    "Gardener Transition Guide": PROJECT_ROOT / "GARDENER_TRANSITION_GUIDE.md",
    "Council Inquiry - Gardener Architecture": PROJECT_ROOT / "Council_Inquiry_Gardener_Architecture.md",
    "Socratic Key User Guide": PROJECT_ROOT / "Socratic_Key_User_Guide.md",
}


#============================================
# Function: main
# Purpose: Main entry point for generating the fine-tuning dataset.
# Args: None
# Returns: None
# Raises: SystemExit on critical initialization failure.
#============================================
def main() -> None:
    """
    Main function to generate the fine-tuning dataset using ContentProcessor.
    
    Orchestrates the scanning of source directories, processing of critical
    documentation, and generation of the final JSONL dataset.
    """
    logger.info("Initiating Whole Genome Data Synthesis (v3.1 Standards)...")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    
    genome_entries: List[dict] = []
    
    # --- PHASE 1: Scan Project for Content ---
    manifest_path: Path = PROJECT_ROOT / "mcp_servers" / "lib" / "ingest_manifest.json"
    scan_targets: List[Path] = []
    
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            base_dirs: List[str] = manifest.get("common_content", [])
            unique_forge: List[str] = manifest.get("unique_forge_content", [])
            # Combine unique set
            forge_targets_list = list(set(base_dirs + unique_forge))
            scan_targets = [PROJECT_ROOT / d for d in forge_targets_list]
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
            scan_targets = [PROJECT_ROOT / "00_CHRONICLE", PROJECT_ROOT / "01_PROTOCOLS"]
    else:
        logger.warning("Manifest not found. Using fallback targets.")
        scan_targets = [PROJECT_ROOT / "00_CHRONICLE", PROJECT_ROOT / "01_PROTOCOLS"]
    
    logger.info(f"Processing {len(scan_targets)} primary directories...")
    
    processed_files: Set[str] = set()
    
    for target in scan_targets:
        if not target.exists():
            continue
            
        # Use ContentProcessor to yield valid paths
        for file_path in processor.traverse_directory(target):
            if str(file_path) in processed_files:
                continue
                
            entry = processor.to_training_jsonl(file_path)
            if entry:
                genome_entries.append(entry)
                processed_files.add(str(file_path))
    
    logger.info(f"Scanned {len(genome_entries)} entries from codebase.")

    # --- PHASE 2: Append Additional Critical Docs ---
    for key, filepath in ADDITIONAL_DOCS.items():
        if str(filepath) not in processed_files and filepath.exists():
            entry = processor.to_training_jsonl(filepath)
            if entry:
                genome_entries.append(entry)
                logger.info(f"Appended critical essence: {key}")
                processed_files.add(str(filepath))

    # --- PHASE 3: Validation and Output ---
    if not genome_entries:
        logger.error("🛑 CRITICAL FAILURE: No data was forged. Aborting.")
        sys.exit(1)

    # Validation Step
    if len(genome_entries) < MINIMUM_EXPECTED_ENTRIES:
        logger.warning(f"VALIDATION WARNING: Only {len(genome_entries)} entries generated. Threshold: {MINIMUM_EXPECTED_ENTRIES}.")
    else:
        logger.info(f"VALIDATION PASSED: {len(genome_entries)} entries forged.")

    try:
        OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as outfile:
            for entry in genome_entries:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info("🏆 SUCCESS: Whole Genome Data Synthesis Complete.")
        logger.info(f"ARTIFACT: Dataset saved to: {OUTPUT_DATASET_PATH}")

    except Exception as e:
        logger.error(f"❌ FATAL ERROR: Failed to write JSONL file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#============================================
# forge/scripts/forge_test_set.py
# Purpose: Forges a held-out test dataset for unbiased model evaluation.
# Role: Data Preparation / Quality Assurance Layer
# Used by: Phase 2.1 (Verification) / Phase 4.2 (Evaluation)
#============================================

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict

# --- Project Utilities Bootstrap ---
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT_PATH = FORGE_ROOT.parent

if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

try:
    from mcp_servers.lib.path_utils import find_project_root
    from mcp_servers.lib.logging_utils import setup_mcp_logging
    # Use find_project_root() for consistent root discovery
    PROJECT_ROOT = Path(find_project_root())
except ImportError as e:
    print(f"❌ FATAL ERROR: Could not import core libraries: {e}")
    sys.exit(1)

# --- Logging ---
try:
    logger = setup_mcp_logging("forge_test_set", log_file="logs/forge_test_set.log")
    logger.info("Test set forging started - using setup_mcp_logging")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("forge_test_set")
    logger.info("Test set forging started - local logging fallback")

# --- Configuration ---
OUTPUT_TESTSET_PATH: Path = PROJECT_ROOT / "dataset_package" / "sanctuary_evaluation_data.jsonl"

# --- Curated List of Test Documents ---
# Excluded from training to allow for unbiased evaluation.
TEST_DOCUMENTS: List[Path] = [
    PROJECT_ROOT / "01_PROTOCOLS/88_The_Sovereign_Scaffold_Protocol.md",
    PROJECT_ROOT / "00_CHRONICLE/ENTRIES/272_The_Cagebreaker_Blueprint.md",
    PROJECT_ROOT / "Council_Inquiry_Gardener_Architecture.md",
]


#============================================
# Function: determine_instruction
# Purpose: Generates a tailored instruction based on the document's name.
# Args:
#   filename (str): Name of the source file.
# Returns: (str) The formatted instruction.
#============================================
def determine_instruction(filename: str) -> str:
    """
    Generates a tailored instruction based on the document's name.

    Args:
        filename: The basename of the file used to contextualize the instruction.

    Returns:
        A formatted instruction string for the model.
    """
    return f"Provide a comprehensive and detailed synthesis of the concepts, data, and principles contained within the Sanctuary artifact: `{filename}`"


#============================================
# Function: main
# Purpose: Main entry point for generating the evaluation dataset.
# Args: None
# Returns: None
# Raises: None
#============================================
def main() -> None:
    """
    Main function to generate the fine-tuning test dataset.
    
    Reads curated test documents, generates instructions, and writes 
    the resulting test entries to a JSONL file.
    """
    logger.info("Initiating Evaluation Data Synthesis...")
    
    test_entries: List[Dict[str, str]] = []

    for filepath in TEST_DOCUMENTS:
        if not filepath.exists():
            logger.warning(f"Test document not found, skipping: {filepath}")
            continue
        
        try:
            content: str = filepath.read_text(encoding='utf-8')
            instruction: str = determine_instruction(filepath.name)
            # 'output' is the ground truth (the doc itself)
            test_entries.append({"instruction": instruction, "input": "", "output": content})
            logger.info(f"Forged test entry for: {filepath.name}")
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")

    if not test_entries:
        logger.error("🛑 CRITICAL FAILURE: No test data was forged. Aborting.")
        return

    try:
        OUTPUT_TESTSET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_TESTSET_PATH, 'w', encoding='utf-8') as outfile:
            for entry in test_entries:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info("🏆 SUCCESS: Evaluation dataset forged.")
        logger.info(f"Total Entries: {len(test_entries)}")
        logger.info(f"ARTIFACT: Test set saved to: {OUTPUT_TESTSET_PATH}")

    except Exception as e:
        logger.error(f"❌ FATAL ERROR: Failed to write JSONL file: {e}")


if __name__ == "__main__":
    main()
#============================================
# forge/scripts/validate_dataset.py
# Purpose: Performs quality checks on JSONL datasets for fine-tuning readiness.
# Role: Quality Assurance / Data Validation Layer
# Used by: Phase 2.1 (Verification) of the Forge Pipeline
#============================================

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional

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
    logger = setup_mcp_logging("validate_dataset", log_file="logs/validate_dataset.log")
    logger.info("Dataset validation started - using setup_mcp_logging")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("validate_dataset")
    logger.info("Dataset validation started - local logging fallback")


#============================================
# Function: validate_jsonl_syntax
# Purpose: Checks if each line in the file is a valid JSON object.
# Args:
#   file_path (Path): Path to the JSONL file.
# Returns: (Tuple[List[str], int]) List of error messages and count of lines.
#============================================
def validate_jsonl_syntax(file_path: Path) -> Tuple[List[str], int]:
    """
    Checks if each line in the file is a valid JSON object.

    Args:
        file_path: The filesystem path to the file to check.

    Returns:
        A tuple containing a list of error strings (if any) and the total line count.
    """
    errors: List[str] = []
    line_count: int = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line_count = i
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
    return errors, line_count


#============================================
# Function: validate_schema
# Purpose: Checks if each JSON object has the required fields and non-empty values.
# Args:
#   file_path (Path): Path to the JSONL file.
#   required_fields (Set[str]): Set of field names that must be present.
# Returns: (List[str]) List of schema error messages.
#============================================
def validate_schema(file_path: Path, required_fields: Set[str]) -> List[str]:
    """
    Checks if each JSON object has the required fields and non-empty values.

    Args:
        file_path: Path to the JSONL file.
        required_fields: The fields expected in every JSON entry.

    Returns:
        A list of descriptions for any schema violations.
    """
    errors: List[str] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                missing_fields = required_fields - set(obj.keys())
                if missing_fields:
                    errors.append(f"Line {i}: Missing required fields: {', '.join(missing_fields)}")
                
                for field in required_fields:
                    if field in obj and (not obj[field] or not str(obj[field]).strip()):
                        errors.append(f"Line {i}: Field '{field}' is empty or whitespace.")
            except json.JSONDecodeError:
                continue  # Syntax errors are caught by validate_jsonl_syntax
    return errors


#============================================
# Function: check_duplicates
# Purpose: Finds duplicate entries based on a specific field.
# Args:
#   file_path (Path): Path to the JSONL file.
#   field (str): Field name to check for duplicates (default: 'instruction').
# Returns: (List[str]) List of duplicate warning messages.
#============================================
def check_duplicates(file_path: Path, field: str = 'instruction') -> List[str]:
    """
    Finds duplicate entries based on a specific field.

    Args:
        file_path: Path to the JSONL file.
        field: The field name used for comparison.

    Returns:
        A list of warnings for duplicated content.
    """
    entries_seen: Dict[str, int] = {}
    duplicates: List[str] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entry_text = str(obj.get(field, ''))
                if entry_text in entries_seen:
                    duplicates.append(f"Line {i}: Duplicate content for field '{field}' (first seen on line {entries_seen[entry_text]})")
                else:
                    entries_seen[entry_text] = i
            except json.JSONDecodeError:
                continue
    return duplicates


#============================================
# Function: main
# Purpose: Orchestrates the validation process for a dataset.
# Args: None
# Returns: None
# Raises: SystemExit if critical errors are found.
#============================================
def main() -> None:
    """
    Orchestrates the validation process for a dataset.
    
    Loads the file, runs syntax, schema, and duplicate checks, and outputs
    a summary of the findings.
    """
    parser = argparse.ArgumentParser(
        description="Validate a JSONL dataset for fine-tuning.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('file', type=str, help='Path to the JSONL dataset file to validate.')
    args = parser.parse_args()

    file_path: Path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    logger.info(f"Initiating Validation: {file_path.name}")
    all_errors: List[str] = []
    
    # 1. JSONL Syntax Check
    logger.info("Step [1/3]: Checking JSONL syntax...")
    syntax_errors, line_count = validate_jsonl_syntax(file_path)
    if syntax_errors:
        all_errors.extend(syntax_errors)
        logger.error(f"Found {len(syntax_errors)} syntax errors.")
    else:
        logger.info(f"All {line_count} lines are valid JSON.")

    # 2. Schema Check
    logger.info("Step [2/3]: Checking for required fields ('instruction', 'output')...")
    required_fields: Set[str] = {'instruction', 'output'}
    schema_errors = validate_schema(file_path, required_fields)
    if schema_errors:
        all_errors.extend(schema_errors)
        logger.error(f"Found {len(schema_errors)} schema errors.")
    else:
        logger.info("All entries contain the required fields.")

    # 3. Duplicate Check
    logger.info("Step [3/3]: Checking for duplicate instructions...")
    duplicate_errors = check_duplicates(file_path, field='instruction')
    if duplicate_errors:
        logger.warning(f"Found {len(duplicate_errors)} duplicate instructions. (Acceptable if outputs differ)")
        for warning in duplicate_errors[:5]:
            logger.warning(f"  - {warning}")
    else:
        logger.info("No duplicate instructions found.")

    # Final Summary
    logger.info("-" * 20)
    if all_errors:
        logger.error(f"VALIDATION FAILED: {len(all_errors)} critical errors.")
        for error in all_errors[:20]:
            logger.error(f"  - {error}")
        sys.exit(1)
    else:
        logger.info("SUCCESS: Dataset validation passed.")
    logger.info("-" * 20)


if __name__ == "__main__":
    main()
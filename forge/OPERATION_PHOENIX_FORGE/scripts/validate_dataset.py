#!/usr/bin/env python3
# ==============================================================================
# VALIDATE_DATASET.PY (v1.0)
#
# This script performs a series of quality checks on a JSONL dataset to ensure
# it's ready for fine-tuning. It validates JSON syntax, schema, duplicates,
# and provides statistics on the data.
#
# Inspired by the 'validate_dataset.py' script from the Smart-Secrets-Scanner project.
#
# Usage:
#   python forge/OPERATION_PHOENIX_FORGE/scripts/validate_dataset.py [path_to_dataset.jsonl]
# ==============================================================================

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

def validate_jsonl_syntax(file_path):
    """Checks if each line in the file is a valid JSON object."""
    errors = []
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line_count = i
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
    return errors, line_count

def validate_schema(file_path, required_fields):
    """Checks if each JSON object has the required fields and non-empty values."""
    errors = []
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
                continue  # Syntax errors are caught by another function
    return errors

def check_duplicates(file_path, field='instruction'):
    """Finds duplicate entries based on a specific field."""
    entries_seen = {}
    duplicates = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entry_text = obj.get(field, '')
                if entry_text in entries_seen:
                    duplicates.append(f"Line {i}: Duplicate content for field '{field}' (first seen on line {entries_seen[entry_text]})")
                else:
                    entries_seen[entry_text] = i
            except json.JSONDecodeError:
                continue
    return duplicates

def main():
    parser = argparse.ArgumentParser(
        description="Validate a JSONL dataset for fine-tuning.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('file', type=str, help='Path to the JSONL dataset file to validate.')
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"🛑 ERROR: File not found: {file_path}")
        sys.exit(1)

    print(f"--- 🧐 Validating Dataset: {file_path.name} ---")
    all_errors = []
    
    # 1. JSONL Syntax Check
    print("\n[1/3] Checking JSONL syntax...")
    syntax_errors, line_count = validate_jsonl_syntax(file_path)
    if syntax_errors:
        all_errors.extend(syntax_errors)
        print(f"❌ Found {len(syntax_errors)} syntax errors.")
    else:
        print(f"✅ All {line_count} lines are valid JSON.")

    # 2. Schema Check
    print("\n[2/3] Checking for required fields ('instruction', 'output')...")
    # For Project Sanctuary, the core fields are 'instruction' and 'output'.
    required_fields = {'instruction', 'output'}
    schema_errors = validate_schema(file_path, required_fields)
    if schema_errors:
        all_errors.extend(schema_errors)
        print(f"❌ Found {len(schema_errors)} schema errors.")
    else:
        print(f"✅ All entries contain the required fields.")

    # 3. Duplicate Check
    print("\n[3/3] Checking for duplicate instructions...")
    duplicate_errors = check_duplicates(file_path, field='instruction')
    if duplicate_errors:
        # These are warnings, not hard errors, but good to know.
        print(f"⚠️  Found {len(duplicate_errors)} duplicate instructions. This may be acceptable if outputs differ.")
        for warning in duplicate_errors[:5]:
            print(f"  - {warning}")
    else:
        print(f"✅ No duplicate instructions found.")

    # Final Summary
    print("\n" + "="*50)
    if all_errors:
        print(f"🛑 VALIDATION FAILED with {len(all_errors)} critical errors.")
        print("Please review the errors below:")
        for error in all_errors[:20]: # Print up to 20 errors
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("🏆 SUCCESS: Dataset validation passed!")
        print("The dataset appears to be well-formatted and ready for fine-tuning.")
    print("="*50)

if __name__ == "__main__":
    main()
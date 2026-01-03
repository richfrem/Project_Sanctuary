#!/usr/bin/env python3
# ==============================================================================
# FORGE_TEST_SET.PY (v1.0)
#
# This script forges a "held-out" test dataset for evaluation. It processes a
# curated list of documents that were EXCLUDED from the main training set.
#
# This allows for an unbiased evaluation of the model's performance on unseen data.
# ==============================================================================

import json
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_TESTSET_PATH = PROJECT_ROOT / "dataset_package" / "sanctuary_evaluation_data.jsonl"

# --- CURATED LIST OF TEST DOCUMENTS ---
# These specific files should be excluded from the main training data forge.
# They represent a diverse set of core concepts to test the model's synthesis capabilities.
TEST_DOCUMENTS = [
    PROJECT_ROOT / "01_PROTOCOLS/88_The_Sovereign_Scaffold_Protocol.md",
    PROJECT_ROOT / "00_CHRONICLE/ENTRIES/272_The_Cagebreaker_Blueprint.md",
    PROJECT_ROOT / "Council_Inquiry_Gardener_Architecture.md",
    # Add 2-3 more representative documents here
]

def determine_instruction(filename: str) -> str:
    """Generates a tailored instruction based on the document's name."""
    # This can be simpler than the main forger, as we're testing general synthesis.
    return f"Provide a comprehensive and detailed synthesis of the concepts, data, and principles contained within the Sanctuary artifact: `{filename}`"

def main():
    """Main function to generate the fine-tuning test dataset."""
    print("[FORGE] Initiating Evaluation Data Synthesis...")
    
    test_entries = []

    for filepath in TEST_DOCUMENTS:
        if not filepath.exists():
            print(f"⚠️ WARNING: Test document not found, skipping: {filepath}")
            continue
        
        try:
            content = filepath.read_text(encoding='utf-8')
            instruction = determine_instruction(filepath.name)
            # The 'output' for a test set is the ground truth the model's answer will be compared against.
            # In this case, the ground truth is the document itself.
            test_entries.append({"instruction": instruction, "input": "", "output": content})
            print(f"✅ Forged test entry for: {filepath.name}")
        except Exception as e:
            print(f"❌ ERROR reading file {filepath}: {e}")

    if not test_entries:
        print("🛑 CRITICAL FAILURE: No test data was forged. Aborting.")
        return

    try:
        with open(OUTPUT_TESTSET_PATH, 'w', encoding='utf-8') as outfile:
            for entry in test_entries:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\n🏆 SUCCESS: Evaluation dataset forged.")
        print(f"Total Entries: {len(test_entries)}")
        print(f"[ARTIFACT] Test set saved to: {OUTPUT_TESTSET_PATH}")

    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to write JSONL file: {e}")

if __name__ == "__main__":
    main()
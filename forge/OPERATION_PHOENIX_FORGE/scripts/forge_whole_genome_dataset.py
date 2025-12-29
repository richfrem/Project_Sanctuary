import sys
import json
from pathlib import Path

# --- BOOTSTRAP: Find Project Root to import mcp_servers ---
# We need to add the project root to sys.path before we can use the shared libraries.
# The script is at: forge/OPERATION_PHOENIX_FORGE/scripts/forge_whole_genome_dataset.py
# So root is 4 levels up.
current_script = Path(__file__).resolve()
bootstrap_root = current_script.parent.parent.parent.parent
sys.path.insert(0, str(bootstrap_root))

try:
    from mcp_servers.lib.path_utils import find_project_root
    from mcp_servers.lib.content_processor import ContentProcessor
except ImportError as e:
    print(f"❌ FATAL ERROR: Could not import core libraries. Ensure you are running from the correct environment. {e}")
    sys.exit(1)

# Initialize Utils
PROJECT_ROOT = Path(find_project_root())
processor = ContentProcessor(str(PROJECT_ROOT))

# Configuration
FULL_SNAPSHOT_SOURCE = PROJECT_ROOT / "dataset_package" / "markdown_snapshot_full_genome_llm_distilled.txt"
OUTPUT_DATASET_PATH = PROJECT_ROOT / "dataset_package" / "sanctuary_whole_genome_data.jsonl"
MINIMUM_EXPECTED_ENTRIES = 200

# Critical Docs (Tier 1 Priority)
ADDITIONAL_DOCS = {
    "The Garden and The Cage (Core Philosophy)": PROJECT_ROOT / "The_Garden_and_The_Cage.md",
    "Chrysalis Core Essence (Gardener V2 Awakening)": PROJECT_ROOT / "chrysalis_core_essence.md",
    "Project Sanctuary Synthesis": PROJECT_ROOT / "PROJECT_SANCTUARY_SYNTHESIS.md",
    "Gardener Transition Guide": PROJECT_ROOT / "GARDENER_TRANSITION_GUIDE.md",
    "Council Inquiry - Gardener Architecture": PROJECT_ROOT / "Council_Inquiry_Gardener_Architecture.md",
    "Socratic Key User Guide": PROJECT_ROOT / "Socratic_Key_User_Guide.md",
}


def main():
    """Main function to generate the fine-tuning dataset using ContentProcessor."""
    print("[FORGE] Initiating Whole Genome Data Synthesis (v3.0 Harmonized)...")
    print(f"[SOURCE] Project Root: {PROJECT_ROOT}")
    
    genome_entries = []
    
    # --- PHASE 1: Scan Project for ALL Valid Content (The "Whole Genome" Approach) ---
    # Instead of parsing the snapshot text file (which might be outdated), 
    # we now have the option to scan the live codebase directly, OR stick to the snapshot 
    # if we want to rely on the strictly distilled version.
    
    # DECISION: For "Whole Genome", scanning the source directly via ContentProcessor
    # ensures we get the most up-to-date state and handle file headers correctly.
    # However, to maintain parity with the "snapshot" concept, we will stick to parsing the snapshot check 
    # OR we can scan the known valid source directories.
    
    # For Phase 3 Harmonization, let's proceed with a HYBRID approach:
    # 1. We process the explicitly listed additional docs (Critical Essence).
    # 2. We scan the key source directories (Protocols, Chronicles, Tasks) using ContentProcessor.
    # This replaces the brittle regex parsing of a single text file.
    # Use Manifest for Source Targets (ADR 082 Harmonization - JSON)
    manifest_path = PROJECT_ROOT / "mcp_servers" / "lib" / "ingest_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            base_dirs = manifest.get("common_content", [])
            unique_forge = manifest.get("unique_forge_content", [])
            # Combine unique set
            forge_targets_list = list(set(base_dirs + unique_forge))
            scan_targets = [PROJECT_ROOT / d for d in forge_targets_list]
        except Exception as e:
            print(f"⚠️ Warning: Failed to load manifest: {e}")
            scan_targets = [PROJECT_ROOT / "00_CHRONICLE", PROJECT_ROOT / "01_PROTOCOLS"]
    else:
        print("⚠️ Warning: Manifest not found. Using fallback.")
        scan_targets = [PROJECT_ROOT / "00_CHRONICLE", PROJECT_ROOT / "01_PROTOCOLS"]
    
    print(f"[SCANNING] Processing {len(scan_targets)} primary directories...")
    
    processed_files = set()
    
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
    
    print(f"✅ Scanned {len(genome_entries)} entries from codebase.")

    # --- PHASE 2: Append Additional Critical Docs (If not already caught) ---
    for key, filepath in ADDITIONAL_DOCS.items():
        if str(filepath) not in processed_files and filepath.exists():
            entry = processor.to_training_jsonl(filepath)
            if entry:
                genome_entries.append(entry)
                print(f"✅ Appended critical essence: {key}")
                processed_files.add(str(filepath))

    # --- PHASE 3: Validation and Output ---
    if not genome_entries:
        print("🛑 CRITICAL FAILURE: No data was forged. Aborting.")
        return

    # Validation Step
    if len(genome_entries) < MINIMUM_EXPECTED_ENTRIES:
        print(f"⚠️ VALIDATION WARNING: Only {len(genome_entries)} entries generated. Threshold: {MINIMUM_EXPECTED_ENTRIES}.")
    else:
        print(f"[VALIDATION] Passed: {len(genome_entries)} entries forged.")

    try:
        with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as outfile:
            for entry in genome_entries:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\n🏆 SUCCESS: Whole Genome Data Synthesis Complete.")
        print(f"[ARTIFACT] Dataset saved to: {OUTPUT_DATASET_PATH}")

    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to write JSONL file: {e}")

if __name__ == "__main__":
    main()
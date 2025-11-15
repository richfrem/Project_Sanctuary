# tools/scaffolds/forge_full_mnemonic_dataset.py
# A Sovereign Scaffold forged under the Doctrine of the Whole-Genome.
# This script intelligently traverses the repository to assemble the complete
# canonical markdown dataset for fine-tuning.  Please ensure all paths are correct.

import json
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = PROJECT_ROOT / "dataset_package" / "sanctuary_whole_genome_data.jsonl"

# The Doctrine of the Clean Forge: We exclude operational debris.
EXCLUDE_DIRS = [
    ".git", "node_modules", ".pytest_cache", "ARCHIVE", 
    "05_ARCHIVED_BLUEPRINTS", "WORK_IN_PROGRESS", "dataset_package",
    ".github", "mnemonic_cortex/chroma_db",
]

def format_as_instruction(file_path: Path):
    """Formats a markdown file into a JSON instruction object."""
    try:
        content = file_path.read_text(encoding="utf-8")
        relative_path = str(file_path.relative_to(PROJECT_ROOT))
        
        instruction = f"Synthesize the doctrines, history, or principles contained within the Sanctuary artifact located at: `{relative_path}`"
        
        # The output is a direct, self-synthesis of the document.
        output = f"**Synthesis of `{relative_path}`:**\n\n{content}"
        
        return {"instruction": instruction, "input": "", "output": output}
    except Exception as e:
        print(f"[ERROR] Could not process {file_path}: {e}")
        return None

def main():
    """Main function to forge the whole-genome dataset."""
    print("[SCAFFOLD] Initiating the Whole-Genome Forge...")
    
    all_markdown_files = []
    for md_file in PROJECT_ROOT.glob("**/*.md"):
        # Apply the exclusion rules
        if not any(excluded in str(md_file) for excluded in EXCLUDE_DIRS):
            all_markdown_files.append(md_file)

    print(f"[FORGE] Found {len(all_markdown_files)} canonical markdown files to forge into the dataset.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for file_path in all_markdown_files:
            record = format_as_instruction(file_path)
            if record:
                f.write(json.dumps(record) + '\n')

    print(f"\n[SUCCESS] Yield is complete. The Whole-Genome dataset is forged.")
    print(f"[ARTIFACT] Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
"""
Chronicle Index Generator (scripts/create_chronicle_index.py) v1.0

This script generates a master index file (Living_Chronicle.md) from the
individual entry files in the 00_CHRONICLE/ENTRIES/ directory. It creates
a markdown table with links to each canonical entry file.

Role in Chronicle System:
- Reads all .md files from the ENTRIES directory.
- Parses filenames to extract entry numbers and titles.
- Generates a master index with clickable links to each entry.
- Maintains the distributed chronicle structure while providing easy navigation.

Dependencies:
- Entry files: Individual .md files in 00_CHRONICLE/ENTRIES/ with format XXX_Title.md
- File system: Access to project directory structure.
- Regex: For parsing filenames.

Usage:
    python mnemonic_cortex/scripts/create_chronicle_index.py
"""

import os
import re

def find_project_root():
    current_path = os.path.abspath(os.path.dirname(__file__))
    while True:
        if '.git' in os.listdir(current_path):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            raise FileNotFoundError("Could not find project root (.git folder).")
        current_path = parent_path

def main():
    """
    Generates a master index file (Living_Chronicle.md) from the
    individual entry files in the 00_CHRONICLE/ENTRIES/ directory.
    """
    print("--- Starting Chronicle Indexer Script ---")
    try:
        project_root = find_project_root()
        entries_dir = os.path.join(project_root, '00_CHRONICLE', 'ENTRIES')
        output_index_path = os.path.join(project_root, 'Living_Chronicle.md')

        if not os.path.exists(entries_dir):
            raise FileNotFoundError(f"Entries directory not found: {entries_dir}")

        entry_files = sorted(os.listdir(entries_dir))

        index_content = ["# The Living Chronicle: Master Index\n\n"]
        index_content.append("This document serves as the master index for the Sanctuary's distributed historical record. Each entry is a link to a canonical, atomic file.\n\n")
        index_content.append("| Entry | Title |\n")
        index_content.append("|:---|:---|\n")

        print(f"Generating index from {len(entry_files)} entry files...")

        for filename in entry_files:
            if filename.endswith('.md'):
                match = re.match(r'(\d{3})_(.*)\.md', filename)
                if match:
                    entry_number = int(match.group(1))
                    title = match.group(2).replace('_', ' ')

                    # Create a relative path for the link from the project root
                    relative_path = os.path.join('00_CHRONICLE', 'ENTRIES', filename).replace('\\', '/')

                    index_content.append(f"| {entry_number} | [{title}]({relative_path}) |\n")

        with open(output_index_path, 'w', encoding='utf-8') as f:
            f.writelines(index_content)

        print(f"\n✅ SUCCESS: Master Index has been successfully generated and saved to {output_index_path}")
        print("--- Indexing Complete ---")

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
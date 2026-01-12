#============================================
# scripts/hf_decorate_readme.py
# Purpose: Prepares the local Hugging Face staging directory for upload.
# Output: Modifies 'hugging_face_dataset_repo/README.md' in-place.
# Role:
#   1. Ensures the Hub-standard directory structure (lineage, data, metadata).
#   2. Augments the README.md with Hub-readable YAML frontmatter.
#      - This metadata explains the dataset's schema and licensing to the Hub's UI.
#      - It creates the link between the 'Human' landing page and the 'Machine' data records.
#      - IMPORTANT: This script understands the 'data/soul_traces.jsonl' schema and maps it 
#        to the 'features' section in the Dataset Card so the Hub can correctly index the soul.
# Scenarios when to run this script:
#   1. Schema Changes: If you add new fields (e.g., semantic_entropy) to the soul records.
#   2. Structural Changes: If you move data/lineage folders or rename the staging directory.
#   3. Metadata Updates: If you want to update tags, license, or the Hub 'Pretty Name'.
# ADR: 081 - Content Harmonization & Integrity
#============================================
import re
import os
import sys
import shutil
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def stage_readme():
    base_dir = Path("hugging_face_dataset_repo")
    # Ensure ADR 081 Structure
    (base_dir / "lineage").mkdir(exist_ok=True, parents=True)
    (base_dir / "data").mkdir(exist_ok=True)
    (base_dir / "metadata").mkdir(exist_ok=True)

    readme_path = base_dir / "README.md"
    
    # 1. Read Content
    content = readme_path.read_text(encoding="utf-8")
    
    # 2. Prepend YAML Metadata (ADR 081)
    # UPDATED: Removed 'dataset_info' block to allow Hugging Face Auto-Discovery
    # to calculate stats (rows/size) automatically.
    yaml_header = """---
license: cc0-1.0
task_categories:
  - text-generation
language:
  - en
tags:
  - reasoning-traces
  - project-sanctuary
  - cognitive-continuity
  - ai-memory
  - llm-training-data
  - metacognition
pretty_name: Project Sanctuary Soul
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/soul_traces.jsonl
---

"""
    # Remove existing header if it exists
    if content.startswith("---"):
        # Find the second occurrence of ---
        second_dash = content.find("---", 3)
        if second_dash != -1:
            content = content[second_dash + 3:]
    
    new_content = yaml_header + content.lstrip()
    
    readme_path.write_text(new_content, encoding="utf-8")
    print(f"Updated {readme_path} with YAML metadata.")
    print("Created ADR 081 directories: lineage/, data/, metadata/")

if __name__ == "__main__":
    stage_readme()
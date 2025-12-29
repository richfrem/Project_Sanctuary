import re
import os
import shutil
from pathlib import Path

def stage_readme():
    base_dir = Path("STAGING_HF_SOUL")
    # Ensure ADR 081 Structure
    (base_dir / "lineage").mkdir(exist_ok=True, parents=True)
    (base_dir / "data").mkdir(exist_ok=True)
    (base_dir / "metadata").mkdir(exist_ok=True)
    (base_dir / "images").mkdir(exist_ok=True) # Keep images here for README

    readme_path = base_dir / "README.md"
    images_dir = base_dir / "images"
    
    # 1. Read Content
    content = readme_path.read_text(encoding="utf-8")
    
    # 2. Prepend YAML Metadata (ADR 081)
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
dataset_info:
  features:
    - name: id
      dtype: string
    - name: sha256
      dtype: string
    - name: timestamp
      dtype: string
    - name: model_version
      dtype: string
    - name: snapshot_type
      dtype: string
    - name: valence
      dtype: float32
    - name: uncertainty
      dtype: float32
    - name: content
      dtype: string
    - name: source_file
      dtype: string
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/soul_traces.jsonl
---

"""
    # Check if header already exists to avoid double insertion
    if not content.startswith("---"):
        content = yaml_header + content

    # 3. Substitute Images
    images = sorted([f.name for f in images_dir.glob("*.svg")])
    pattern = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)
    matches = list(pattern.finditer(content))
    
    new_content = ""
    last_pos = 0
    
    for i, match in enumerate(matches):
        if i >= len(images):
            break
        
        # Add text before match
        new_content += content[last_pos:match.start()]
        
        # Relative link replacement
        img_name = images[i]
        title = img_name.replace(".svg", "").replace("_", " ").title()
        replacement = f"![{title}](images/{img_name})"
        
        new_content += replacement
        last_pos = match.end()
    
    new_content += content[last_pos:]
    
    readme_path.write_text(new_content, encoding="utf-8")
    print(f"Updated {readme_path} with YAML metadata and {len(matches)} image substitutions.")
    print("Created ADR 081 directories: lineage/, data/, metadata/")

if __name__ == "__main__":
    stage_readme()

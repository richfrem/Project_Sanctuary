import re
import os
from pathlib import Path

def prepare_readme():
    project_root = Path(os.getcwd())
    readme_path = project_root / "README.md"
    images_dir = project_root / "README-MARKDOWN-IMAGES"
    output_path = project_root / "README_HF.md"
    
    # 1. Read Original README
    content = readme_path.read_text(encoding="utf-8")
    
    # 2. Get Ordered Image List
    images = sorted([f.name for f in images_dir.glob("*.svg")])
    if not images:
        print("No images found!")
        return
        
    print(f"Found {len(images)} images for replacement.")
    
    # 3. Find Mermaid Blocks
    # Regex to find fenced code blocks with 'mermaid' language
    pattern = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)
    
    matches = list(pattern.finditer(content))
    print(f"Found {len(matches)} mermaid blocks in README.md.")
    
    if len(matches) != len(images):
        print(f"WARNING: Mismatch count! Mermaid blocks: {len(matches)}, Images: {len(images)}")
        # Proceeding with min length to avoid index errors, but alerting user
    
    # 4. Perform Replacement
    # We rebuild the string to handle offsets correctly
    new_content = ""
    last_pos = 0
    
    for i, match in enumerate(matches):
        if i >= len(images):
            break
            
        # Append text before this block
        new_content += content[last_pos:match.start()]
        
        # Create Image Link
        # Assuming we upload images to an 'images/' folder in HF
        image_filename = images[i]
        diagram_title = image_filename.replace(".svg", "").replace("_", " ").title()
        image_link = f"![{diagram_title}](https://huggingface.co/datasets/richfrem/Project_Sanctuary_Soul/resolve/main/images/{image_filename})"
        
        new_content += image_link
        
        last_pos = match.end()
    
    # Append remaining text
    new_content += content[last_pos:]
    
    # 5. Write Output
    output_path.write_text(new_content, encoding="utf-8")
    print(f"Generated {output_path}")

if __name__ == "__main__":
    prepare_readme()

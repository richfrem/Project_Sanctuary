import json
import hashlib
from datetime import datetime
from pathlib import Path
from mcp_servers.lib.content_processor import ContentProcessor

def generate_data():
    project_root = Path.cwd()
    staging_dir = project_root / "STAGING_HF_SOUL"
    data_dir = staging_dir / "data"
    
    # Ensure structure (no lineage folder needed)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    processor = ContentProcessor(str(project_root))
    
    # Allow-list for root-level files (everything else at root is excluded)
    ROOT_ALLOW_LIST = {
        "README.md",
        "chrysalis_core_essence.md",
        "Council_Inquiry_Gardener_Architecture.md",
        "Living_Chronicle.md",
        "PROJECT_SANCTUARY_SYNTHESIS.md",
        "Socratic_Key_User_Guide.md",
        "The_Garden_and_The_Cage.md",
        "GARDENER_TRANSITION_GUIDE.md",
    }
    
    records = []
    
    print("🧠 Generating Soul Data...")
    
    # Traverse project
    for file_path in processor.traverse_directory(project_root):
        try:
            rel_path = file_path.relative_to(project_root)
        except ValueError:
            continue
            
        # Filter out STAGING_HF_SOUL itself
        if str(rel_path).startswith("STAGING_HF_SOUL"):
            continue
        
        # Root-level file filter: only allow explicit list
        if rel_path.parent == Path("."):
            if rel_path.name not in ROOT_ALLOW_LIST:
                continue
        
        try:
            # Read and transform content directly (no intermediate files)
            content = processor.transform_to_markdown(file_path)
            content_bytes = content.encode('utf-8')
            checksum = hashlib.sha256(content_bytes).hexdigest()
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Clean ID from relative path
            clean_id = str(rel_path).replace("/", "_").replace("\\", "_")
            # Strip .md extensions
            while clean_id.endswith('.md'):
                clean_id = clean_id[:-3]
            
            record = {
                "id": clean_id,
                "sha256": checksum,
                "timestamp": timestamp,
                "model_version": "Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
                "snapshot_type": "seal",
                "valence": 0.5,
                "uncertainty": 0.1,
                "content": content,
                "source_file": str(rel_path)
            }
            records.append(record)
            
        except Exception as e:
            print(f"Skipping {rel_path}: {e}")
            
    # Write JSONL
    jsonl_path = data_dir / "soul_traces.jsonl"
    print(f"📝 Writing {len(records)} records to {jsonl_path}")
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            
    print("✅ Soul Data Generation Complete.")

if __name__ == "__main__":
    generate_data()

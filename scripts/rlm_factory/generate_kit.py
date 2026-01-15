#!/usr/bin/env python3
import os
import json
from pathlib import Path
from datetime import datetime

"""
RLM Bootstrap Kit Generator
---------------------------
Packages the Reactive Ledger Memory (RLM) pattern, code, and documentation 
into a single "Prompt-Ready" artifact for replication in other repositories.
"""

def read_file(path: Path) -> str:
    if not path.exists():
        return f"[ERROR: File not found: {path}]"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_kit():
    # Adjusted: We are in scripts/rlm_factory/, so root is ../../..
    project_root = Path(__file__).parent.parent.parent.absolute()
    # 1. Load the Manifest
    manifest_path = Path(__file__).parent / "kit_manifest.json"
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    output_file = project_root / manifest.get("output_filename", "RLM_RELEASE_PACKET.md")
    
    # 2. Build the Header
    content = [
        "# RLM Bootstrap Packet: The Reactive Ledger Memory Kit",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Description:** {manifest.get('description', 'Auto-generated RLM Kit')}",
        "",
        "## 🚀 Instructions for the AI Agent",
        "You are receiving a specialized architectural pattern called 'Reactive Ledger Memory' (RLM).",
        "Your task is to implement this system in the current repository to give it 'Holographic Memory'.",
        "1.  **Read the Blueprint:** Understand the theory in Section 1.",
        "2.  **Install Dependencies:** `pip install requests genericpath python-dotenv` (plus `ollama` locally).",
        "3.  **Adapt the Code:** The provided Python code is from 'Project Sanctuary'. Adapt paths/imports to fit THIS project's structure.",
        "4.  **Initialize:** Run the distillation loop on the `docs/` folder first.",
        "",
        "---",
        ""
    ]

    # 3. Inject Assets from Manifest
    for component in manifest["components"]:
        source_path = project_root / component["source_path"]
        
        content.append(f"## 📦 Component: {component['title']}")
        content.append(f"**Source:** `{component['source_path']}`")
        if "description" in component:
            content.append(f"_{component['description']}_")
        content.append("")
        
        content.append(f"```{component.get('language', 'markdown')}")
        content.append(read_file(source_path))
        content.append("```")
        content.append("")
        content.append("---")
        content.append("")

    # 4. Inject Environment Setup (The "Glue")
    content.append("## 🛠️ Environment Setup Guide")
    content.append("### 1. Configuration (.env)")
    content.append("Create a `.env` file in your project root:")
    content.append("```bash")
    content.append("RLM_CACHE_PATH=.agent/learning/rlm_summary_cache.json")
    content.append("RLM_TARGET_DIRS=src,docs,lib  # Customize this list for your repo")
    content.append("```")
    content.append("")
    content.append("### 2. Prerequisite: Ollama (The Brain)")
    content.append("1.  **Install:** `curl -fsSL https://ollama.com/install.sh | sh`")
    content.append("2.  **Pull Model:** `ollama pull qwen2.5:7b` (Recommended default)")
    content.append("3.  **Serve:** `ollama serve`")
    content.append("")
    content.append("### Prerequisite: Vector DB (Optional Augmentation)")
    content.append("If augmenting a Vector DB (Chroma), ensure you prepend the RLM summary to chunks:")
    content.append("```python")
    content.append("# Pseudo-code for Augmentation")
    content.append("summary = ledger[file_path]['summary']")
    content.append("chunk_content = f\"[CONTEXT_SUMMARY: {summary}]\\n\\n{original_chunk}\"")
    content.append("vector_db.add(documents=[chunk_content])")
    content.append("```")

    # 5. Write to Disk
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    
    print(f"✅ RLM Bootstrap Packet generated at: {output_file}")
    print(f"📦 Size: {output_file.stat().st_size / 1024:.2f} KB")

if __name__ == "__main__":
    generate_kit()

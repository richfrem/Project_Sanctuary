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
        "# RLM Bootstrap Packet: The Cognitive Architecture Kit (RLM + RAG)",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Description:** {manifest.get('description', 'Auto-generated RLM Kit')}",
        "",
        "## 🚀 Instructions for the AI Agent",
        "You are receiving a complete 'Cognitive Architecture' starter kit.",
        "It contains two complimentary systems:",
        "1.  **Semantic Ledger (RLM):** High-level summaries of every file (`operations.py`).",
        "2.  **Vector Memory (RAG):** Deep retrieval of code chunks (`chroma_ingest.py`).",
        "",
        "**Action Plan:**",
        "1.  **Install:** `pip install requests genericpath python-dotenv langchain-chroma langchain-huggingface`.",
        "2.  **Digest:** Read `REACTIVE_LEDGER_MEMORY.md` (Theory) and `operations.py` (Logic).",
        "3.  **Boot:** Run `python chroma_ingest.py ingest src/` to hydrate your vector memory.",
        "4.  **Init:** Run the RLM distillation loop to build your semantic ledger.",
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

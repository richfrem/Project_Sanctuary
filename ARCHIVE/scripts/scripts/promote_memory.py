#!/usr/bin/env python3
"""
Agentic OS - Memory Promotion Script (Dementia Guard)
Scans dated session logs in context/memory/ and promotes new facts to
the context/memory.md ledger with persistent IDs [#NNN].
"""
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path

def get_fact_id(ledger_content):
    """Extracts the highest [#NNN] ID from the ledger."""
    ids = re.findall(r"\[#(\d+)\]", ledger_content)
    if not ids:
        return 0
    return max(int(i) for i in ids)

def get_content_hashes(ledger_content):
    """Simple hash of lines to avoid exact duplicates."""
    hashes = set()
    for line in ledger_content.splitlines():
        if line.strip():
            hashes.add(hashlib.md5(line.strip().encode()).hexdigest())
    return hashes

def main():
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    memory_dir = Path(project_dir) / "context" / "memory"
    ledger_file = Path(project_dir) / "context" / "memory.md"
    
    if not ledger_file.exists():
        ledger_file.write_text("# Project Memory Ledger\n\n", encoding="utf-8")
        
    ledger_content = ledger_file.read_text(encoding="utf-8")
    current_id = get_fact_id(ledger_content)
    existing_hashes = get_content_hashes(ledger_content)
    
    new_facts = []
    
    # Scan all markdown files in context/memory/
    if memory_dir.exists():
        for file_path in memory_dir.glob("*.md"):
            # Avoid self or ledger
            if file_path.name == "memory.md":
                continue
                
            content = file_path.read_text(encoding="utf-8")
            # Look for Fact blocks: ## [Topic] or ## [Fact]
            # Pattern: ## [(.*?)]\n(.*?)(?=\n##|\Z)
            matches = re.findall(r"## \[(.*?)\]\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
            
            for topic, body in matches:
                topic = topic.strip()
                body = body.strip()
                
                # Simple deduplication check
                content_hash = hashlib.md5(body.encode()).hexdigest()
                if content_hash in existing_hashes:
                    continue
                    
                current_id += 1
                date_str = datetime.now().strftime("%Y-%m-%d")
                fact_entry = f"## [{date_str}] [#{current_id:03d}] [{topic}]\n{body}\n"
                new_facts.append(fact_entry)
                existing_hashes.add(content_hash)
                
    if new_facts:
        print(f"Promoting {len(new_facts)} new facts to the ledger...")
        with open(ledger_file, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(new_facts) + "\n")
    else:
        print("No new unique facts to promote.")

if __name__ == "__main__":
    main()

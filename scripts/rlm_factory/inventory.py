import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_inventory():
    # Adjusted for location in scripts/rlm_factory/
    project_root = Path(__file__).parent.parent.parent.absolute()
    # Load from Env or Default
    env_cache = os.getenv("RLM_CACHE_PATH")
    if env_cache:
        cache_path = project_root / env_cache
    else:
        cache_path = project_root / ".agent" / "learning" / "rlm_summary_cache.json"

    env_targets = os.getenv("RLM_TARGET_DIRS")
    if env_targets:
        target_dirs = [t.strip() for t in env_targets.split(",") if t.strip()]
    else:
        # Default fallback for this repo
        target_dirs = ["docs", "ADRs", "01_PROTOCOLS", "mcp_servers", "LEARNING"]
    
    if not cache_path.exists():
        print(f"❌ Cache not found at {cache_path}")
        return

    with open(cache_path, "r") as f:
        cache = json.load(f)
    
    print(f"🧠 RLM Semantic Ledger Inventory")
    print(f"==============================")
    print(f"{'Directory':<15} | {'Total':<5} | {'Cached':<6} | {'Status':<10}")
    print(f"--------------------------------------------------")

    missing_files = []

    for d in target_dirs:
        dir_path = project_root / d
        if not dir_path.exists():
            continue
            
        # Find all .md and .txt files (as per RLM rules)
        files = []
        for ext in ["**/*.md", "**/*.txt"]:
            files.extend([p for p in dir_path.glob(ext) if p.is_file()])
        
        # Filter out anything in 'archive' subdirs
        files = [f for f in files if "archive" not in str(f).lower()]
        
        total = len(files)
        cached_count = 0
        dir_missing = []

        for f in files:
            rel_path = str(f.relative_to(project_root))
            if rel_path in cache:
                cached_count += 1
            else:
                dir_missing.append(rel_path)
        
        status = "✅ COMPLETE" if total == cached_count else f"⚠️  {total - cached_count} MISSING"
        print(f"{d:<15} | {total:<5} | {cached_count:<6} | {status}")
        
        missing_files.extend(dir_missing)

    if missing_files:
        print(f"\n📑 Missing Files (Ready for targeting):")
        print(f"--------------------------------------")
        # Group by folder for readability
        current_folder = ""
        for mf in sorted(missing_files):
            folder = os.path.dirname(mf)
            if folder != current_folder:
                print(f"\n📂 {folder}/")
                current_folder = folder
            print(f"  - {mf}")
        
        print(f"\n💡 Tip: Run 'python3 scripts/cortex_cli.py rlm-distill <file_path>' to fix specific gaps.")
    else:
        print(f"\n🎉 All priority directories are fully distilled into the Semantic Ledger.")

if __name__ == "__main__":
    get_inventory()

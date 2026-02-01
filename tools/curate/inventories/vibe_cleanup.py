import os
import json
import subprocess
import glob

# Constants
INVENTORY_PATH = "tools/tool_inventory.json"
MANAGER_PATH = "tools/curate/inventories/manage_tool_inventory.py"

def get_missing_files():
    """Run audit and parse missing files from manage_tool_inventory.py output."""
    print("🔍 Auditing inventory for missing files...")
    try:
        result = subprocess.run(
            ["python3", MANAGER_PATH, "audit"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.splitlines()
        missing = []
        is_missing_section = False
        for line in lines:
            if "❌ MISSING FILES" in line:
                is_missing_section = True
                continue
            if is_missing_section:
                if line.startswith("   - "):
                    missing.append(line.replace("   - ", "").strip())
                elif line.strip() == "":
                    continue
                else:
                    # Next section (tracked/untracked) starts
                    if line.strip() and not line.startswith("   -"):
                        is_missing_section = False
        return missing
    except subprocess.CalledProcessError as e:
        print(f"Error running audit: {e.stdout}\n{e.stderr}")
        return []

def remove_tool(path):
    """Remove a tool from the inventory using the manager script."""
    # print(f"🗑️ Removing: {path}")
    try:
        # Use --inventory flag if we wanted to be explicit, but it defaults correctly
        subprocess.run(
            ["python3", MANAGER_PATH, "remove", "--path", path],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error removing {path}: {e.stderr}")

def main():
    missing = get_missing_files()
    
    if not missing:
        print("✅ No missing files found in inventory.")
    else:
        print(f"🗑️ Found {len(missing)} missing files in inventory. Starting removal...")
        for i, path in enumerate(missing):
            print(f"[{i+1}/{len(missing)}] Removing {path}...")
            remove_tool(path)
        print("✨ Pruning complete.")

    # Second phase: Inventory of all current scripts
    print("\n📝 Current Script Inventory (.py, .js):")
    scripts = []
    for ext in ["**/*.py", "**/*.js"]:
        scripts.extend(glob.glob(os.path.join("tools", ext), recursive=True))
    
    for script in sorted(scripts):
        print(f" - {script}")

if __name__ == "__main__":
    main()

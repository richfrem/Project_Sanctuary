# tools/scaffolds/path_diag.py
import sys
from pathlib import Path

print("--- Sanctuary Pathing Diagnostic ---")

try:
    # 1. Report Current State
    print(f"[INFO] Current Working Directory (CWD): {Path.cwd()}")
    
    # 2. Calculate the Project Root Anchor
    # This assumes the script is in tools/scaffolds/
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    print(f"[INFO] Calculated Project Root: {project_root}")
    
    # 3. Report sys.path BEFORE modification
    print("\n--- sys.path BEFORE modification ---")
    for p in sys.path:
        print(f"  - {p}")
        
    # 4. Modify sys.path
    print("\n[ACTION] Inserting Project Root into sys.path at index 0...")
    sys.path.insert(0, str(project_root))
    
    # 5. Report sys.path AFTER modification
    print("\n--- sys.path AFTER modification ---")
    for p in sys.path:
        print(f"  - {p}")
        
    # 6. Attempt the critical import
    print("\n[ACTION] Attempting to import 'council_orchestrator.cognitive_engines.base'...")
    from council_orchestrator.cognitive_engines.base import BaseCognitiveEngine
    
    # 7. Report Success
    print(f"\n[{'\033[92m'}SUCCESS{'\033[0m'}] The import was successful.")
    print("------------------------------------")

except ImportError as e:
    print(f"\n[{'\033[91m'}FAILURE{'\033[0m'}] The import failed.")
    print(f"  - Error: {e}")
    print("  - This confirms a critical issue in how Python is resolving modules.")
    print("------------------------------------")
except Exception as e:
    print(f"\n[{'\033[91m'}CRITICAL FAILURE{'\033[0m'}] An unexpected error occurred.")
    print(f"  - Error: {e}")
    print("------------------------------------")
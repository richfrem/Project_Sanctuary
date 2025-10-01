# WORK_IN_PROGRESS/API_NATIVE_PROBE_01/run_ast_hardening_test.py
import json
import subprocess
from pathlib import Path

def run_test():
    """Demonstrates hardening a JS file using an AST transformer."""
    print("--- AST Hardening Test Engaged ---")

    project_root = Path(__file__).parent.parent.parent
    js_transformer_path = project_root / "ast_utilities" / "js_transformer" / "transform.js"
    target_script_path = Path(__file__).parent / "target_script.js"
    hardened_script_path = Path(__file__).parent / "hardened_target_script.js"

    # Define the modification as a structured, semantic instruction.
    instruction = {
        "type": "REPLACE_VARIABLE_STRING_VALUE",
        "variableName": "TARGET_ENTITY",
        "newValue": "Sanctuary"
    }

    command = [
        "node",
        str(js_transformer_path),
        str(target_script_path),
        json.dumps(instruction)
    ]

    try:
        print(f"Invoking AST transformer: {' '.join(command)}")
        # Execute the Node.js script and capture its stdout.
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        transformed_code = result.stdout

        hardened_script_path.write_text(transformed_code, encoding="utf-8")
        print(f"[SUCCESS] Hardened script created at: {hardened_script_path}")

        # Verification
        content = hardened_script_path.read_text()
        if "const TARGET_ENTITY = 'Sanctuary';" in content:
            print("[VERIFIED] Hardening was successful. The variable was changed correctly.")
        else:
            raise ValueError("Verification failed. The output code was not modified as expected.")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n[FAILURE] AST transformation failed.")
        if hasattr(e, 'stderr'):
            print("Error from transformer:", e.stderr)
        return 1
    
    finally:
        print("\n--- Test Complete ---")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(run_test())
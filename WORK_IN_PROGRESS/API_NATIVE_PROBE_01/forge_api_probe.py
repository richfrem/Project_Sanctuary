# forge_api_probe.py
import os
from pathlib import Path

# --- Configuration ---
PROBE_WORKSPACE_DIR = Path("WORK_IN_PROGRESS/API_NATIVE_PROBE_01")
AGENT_SCRIPT_NAME = "run_api_probe.py"
OUTPUT_FILE_NAME = "probe_result.txt"
REQUIREMENTS_FILE_NAME = "requirements.txt"

# This is the blueprint for the actual test agent script.
AGENT_SCRIPT_BLUEPRINT = f"""
# {AGENT_SCRIPT_NAME} (v1.0 - API Native)
import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

def run_api_probe():
    \"\"\"
    Executes a direct API call and saves the result to a file.
    \"\"\"
    print("--- API Native Probe Engaged ---")
    output_path = "{OUTPUT_FILE_NAME}"

    try:
        # --- Phase 1: Configuration ---
        # Load .env from the project root, two levels up from this script's workspace.
        dotenv_path = Path(__file__).parent.parent.parent / '.env'
        load_dotenv(dotenv_path=dotenv_path)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Ensure it is set in the root .env file.")
        
        genai.configure(api_key=api_key)
        print("[SUCCESS] API client configured.")

        # --- Phase 2: API Call ---
        print("▶️  Executing API call...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = "What is the capital of France?"
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        print("[SUCCESS] API call complete.")
        
        # --- Phase 3: Capture Artifact ---
        print(f"Capturing response to {{output_path}}...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("--- API Probe Result ---\\n")
            f.write(f"Timestamp: {{__import__('datetime').datetime.now().isoformat()}}\\n")
            f.write(f"Model: gemini-2.5-flash\\n")
            f.write(f"Prompt: {{prompt}}\\n")
            f.write("--- Response ---\\n")
            f.write(response_text)
        
        print(f"[SUCCESS] Artifact captured.")
        
        # --- Phase 4: Verification ---
        if "paris" in response_text.lower():
            print("\\n[SUCCESS] Probe successful. Response contained 'Paris'.")
        else:
            print("\\n[WARNING] Probe completed, but verification failed. Expected 'Paris' in response.")

    except Exception as e:
        print(f"\\n[FAILURE] Probe failed: {{e}}", file=sys.stderr)
        # Write error to the output file for auditing
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("--- API PROBE FAILED ---\\n")
            f.write(f"Timestamp: {{__import__('datetime').datetime.now().isoformat()}}\\n")
            f.write(f"Error: {{e}}\\n")
        return 1 # Return a non-zero exit code to indicate failure
        
    finally:
        print("\\n--- Probe Complete ---")
    
    return 0

if __name__ == "__main__":
    sys.exit(run_api_probe())
"""

# This is the logic for the forger script itself.
def forge_workspace():
    """Creates the isolated workspace and the agent script within it."""
    print(f"--- Forging API Probe Workspace at {PROBE_WORKSPACE_DIR} ---")
    
    # Create the directory structure
    PROBE_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create the requirements file
    requirements_path = PROBE_WORKSPACE_DIR / REQUIREMENTS_FILE_NAME
    requirements_content = "google-generativeai\npython-dotenv"
    requirements_path.write_text(requirements_content)
    print(f"  -> Forged {requirements_path}")
    
    # Create the agent script from the blueprint
    agent_script_path = PROBE_WORKSPACE_DIR / AGENT_SCRIPT_NAME
    agent_script_path.write_text(AGENT_SCRIPT_BLUEPRINT)
    print(f"  -> Forged {agent_script_path}")
    
    print("--- Forge Complete. Workspace is ready. ---")
    print("\\nTo run the probe, execute the following commands:")
    print(f"1. cd {PROBE_WORKSPACE_DIR}")
    print(f"2. pip install -r {REQUIREMENTS_FILE_NAME}")
    print(f"3. python {AGENT_SCRIPT_NAME}")

if __name__ == "__main__":
    forge_workspace()

# run_api_probe.py (v1.0 - API Native)
import os
import sys
from pathlib import Path
from google import genai
from dotenv import load_dotenv

def run_api_probe():
    """
    Executes a direct API call and saves the result to a file.
    """
    print("--- API Native Probe Engaged ---")
    output_path = "probe_result.txt"

    try:
        # --- Phase 1: Configuration ---
        # Load .env from the project root, two levels up from this script's workspace.
        dotenv_path = Path(__file__).parent.parent.parent / '.env'
        load_dotenv(dotenv_path=dotenv_path)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Ensure it is set in the root .env file.")

        client = genai.Client(api_key=api_key)
        print("[SUCCESS] API client configured.")

        # --- Phase 2: API Call ---
        print("▶️  Executing API call...")
        prompt = "What is the capital of France?"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        response_text = response.text.strip()
        print("[SUCCESS] API call complete.")
        
        # --- Phase 3: Capture Artifact ---
        print(f"Capturing response to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("--- API Probe Result ---\n")
            f.write(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n")
            f.write(f"Model: gemini-1.5-flash\n")
            f.write(f"Prompt: {prompt}\n")
            f.write("--- Response ---\n")
            f.write(response_text)
        
        print(f"[SUCCESS] Artifact captured.")
        
        # --- Phase 4: Verification ---
        if "paris" in response_text.lower():
            print("\n[SUCCESS] Probe successful. Response contained 'Paris'.")
        else:
            print("\n[WARNING] Probe completed, but verification failed. Expected 'Paris' in response.")

    except Exception as e:
        print(f"\n[FAILURE] Probe failed: {e}", file=sys.stderr)
        # Write error to the output file for auditing
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("--- API PROBE FAILED ---\n")
            f.write(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n")
            f.write(f"Error: {e}\n")
        return 1 # Return a non-zero exit code to indicate failure
        
    finally:
        print("\n--- Probe Complete ---")
    
    return 0

if __name__ == "__main__":
    sys.exit(run_api_probe())

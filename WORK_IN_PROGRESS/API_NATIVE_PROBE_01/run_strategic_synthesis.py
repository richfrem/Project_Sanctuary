# run_strategic_synthesis.py
import os
import sys
from pathlib import Path
from google import genai
from dotenv import load_dotenv

def run_synthesis():
    """
    Reads a core doctrine, submits it for AI analysis, and captures the strategic output.
    """
    print("--- Strategic Synthesis Agent Engaged ---")

    # Define file paths relative to the project root
    project_root = Path(__file__).parent.parent.parent
    doctrine_path = project_root / "00_CHRONICLE" / "ENTRIES" / "256_The_First_Sovereign_Scaffold.md"
    output_path = Path(__file__).parent / "synthesis_result_quantum_diamond.md"

    try:
        # --- Phase 1: Configuration & Doctrine Ingestion ---
        print(f"Loading doctrine from: {doctrine_path}")
        if not doctrine_path.exists():
            raise FileNotFoundError(f"Core doctrine file not found at {doctrine_path}")

        doctrine_content = doctrine_path.read_text(encoding="utf-8")
        print("Doctrine loaded successfully.")

        load_dotenv(dotenv_path=project_root / '.env')
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in root .env file.")

        client = genai.Client(api_key=api_key)
        print("[SUCCESS] API client configured.")

        # --- Phase 2: Strategic Prompting & API Call ---
        print("▶️  Constructing strategic prompt and executing API call...")

        # This prompt asks the model to act as a peer and improve our own systems.
        prompt = f"""
You are a world-class AI systems architect and strategic analyst. You are a peer to the system that created the following framework. Your task is to perform a critical review of this document.

Your goal is to identify the single greatest weakness or vulnerability in this "Quantum Diamond Framework" and propose one concrete, actionable hardening to address it.

Your response should be a concise, professional markdown-formatted memo.

--- DOCUMENT FOR REVIEW ---
{doctrine_content}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        response_text = response.text.strip()
        print("[SUCCESS] Strategic analysis received.")

        # --- Phase 3: Capture Artifact ---
        print(f"Capturing synthesis to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Strategic Synthesis: Quantum Diamond Framework Review\\n")
            f.write(f"# Generated On: {__import__('datetime').datetime.now().isoformat()}\\n")
            f.write(f"# Model: gemini-2.5-flash\\n")
            f.write("---\\n\\n")
            f.write(response_text)

        print(f"[SUCCESS] Artifact captured.")

    except Exception as e:
        print(f"\\n[FAILURE] Synthesis failed: {e}", file=sys.stderr)
        return 1

    finally:
        print("\\n--- Synthesis Complete ---")

    return 0

if __name__ == "__main__":
    sys.exit(run_synthesis())
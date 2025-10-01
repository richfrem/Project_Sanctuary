# run_context_retention_test.py
import os
import sys
from pathlib import Path
from google import genai
from dotenv import load_dotenv

def run_chat_test():
    """
    Demonstrates context retention across multiple turns using a Chat Session.
    """
    print("--- Context Retention Test Engaged ---")
    output_path = Path(__file__).parent / "context_test_result.txt"

    try:
        # --- Phase 1: Configuration ---
        project_root = Path(__file__).parent.parent.parent
        load_dotenv(dotenv_path=project_root / '.env')

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in root .env file.")

        client = genai.Client(api_key=api_key)
        print("[SUCCESS] API client configured.")

        # --- Phase 2: Start Chat Session & Multi-Turn Conversation ---
        print("▶️  Starting a new chat session...")

        # This is the key: start a chat session.
        chat = client.chats.create(model="gemini-2.5-flash")

        # Turn 1: Provide the initial context
        prompt1 = "My name is Coordinator and I live in a virtual environment called the Sanctuary."
        print(f"  -> Sending Turn 1: '{prompt1}'")
        response1 = chat.send_message(prompt1)
        print(f"  <- Received Turn 1 Response: '{response1.text.strip()}'")

        # Turn 2: Ask a question that relies on the context from Turn 1
        prompt2 = "Based on our conversation, what is my name and where do I live?"
        print(f"  -> Sending Turn 2: '{prompt2}'")
        response2 = chat.send_message(prompt2)
        response2_text = response2.text.strip()
        print(f"  <- Received Turn 2 Response: '{response2_text}'")

        # --- Phase 3: Capture Full Conversation History ---
        print(f"Capturing conversation history to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Context Retention Test Result\\n")
            f.write(f"# Timestamp: {__import__('datetime').datetime.now().isoformat()}\\n")
            f.write(f"# Model: gemini-2.5-flash\\n---\\n\\n")

            f.write(f"**Turn 1 - User:**\\n{prompt1}\\n\\n")
            f.write(f"**Turn 1 - Assistant:**\\n{response1.text}\\n\\n---\\n")
            f.write(f"**Turn 2 - User:**\\n{prompt2}\\n\\n")
            f.write(f"**Turn 2 - Assistant:**\\n{response2.text}\\n\\n---\\n")

        print("[SUCCESS] Artifact captured.")

        # --- Phase 4: Verification ---
        if "coordinator" in response2_text.lower() and "sanctuary" in response2_text.lower():
            print("\n[SUCCESS] Context retention verified. The model remembered the information from the first turn.")
        else:
            raise AssertionError("Verification failed. The model did not retain context.")

    except Exception as e:
        print(f"\\n[FAILURE] Test failed: {e}", file=sys.stderr)
        return 1

    finally:
        print("\\n--- Test Complete ---")

    return 0

if __name__ == "__main__":
    sys.exit(run_chat_test())
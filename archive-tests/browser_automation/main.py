# mcp_agent/main.py (v2.0 - HITL Authentication)
import asyncio
from playwright.async_api import async_playwright

# Import the ChatPage object, as LoginPage is now bypassed.
from page_objects.pages.chat_page import ChatPage

# --- Configuration ---
# This URL is the key. It takes us directly to a new chat with the correct model pre-selected.
TARGET_URL = "https://aistudio.google.com/prompts/new_chat?model=gemini-2.5-flash"

# A selector to verify that the manual login was successful and the chat page is ready.
CHAT_PAGE_READY_SELECTOR = 'textarea[placeholder*="logos and brand swag"]'

async def run_hitl_interaction_test():
    """
    Executes an interaction test using a Human-in-the-Loop (HITL) authentication step.
    """
    print("--- MCP Foundational Test: HITL Authentication & Interaction ---")

    async with async_playwright() as p:
        # --- Phase 1: Connect to Steward's Authenticated Browser ---

        print("Connecting to Steward's authenticated Chrome instance at localhost:9222...")
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:9222/json") as resp:
                data = await resp.json()
                ws_url = data[0]['webSocketDebuggerUrl']  # Use the first page's WS URL
        browser = await p.chromium.connect_over_cdp(ws_url)
        # Get the first available page (assuming the Steward has the chat page open)
        pages = browser.contexts[0].pages
        if pages:
            page = pages[0]  # Use the first page
        else:
            raise Exception("No pages found in the connected browser. Please ensure a page is open in the Chrome instance.")

        # --- Phase 2: Autonomous Operation ---

        try:
            print("▶️  Starting autonomous operation...")
            # Verify that the page is ready by checking for a key page element.
            await page.wait_for_selector(CHAT_PAGE_READY_SELECTOR, timeout=15000)
            print("[SUCCESS] Chat page is ready.")

            chat_page = ChatPage(page)

            prompt = "What is the capital of France?"
            response = await chat_page.submit_prompt_and_get_response(prompt)

            # --- Phase 3: Verification ---
            if "paris" in response.lower():
                print("\n[SUCCESS] End-to-end CDP Connect test passed. Response contained 'Paris'.")
            else:
                raise AssertionError(f"Verification failed. Expected 'Paris', got: '{response}'")

        except Exception as e:
            print(f"\n[FAILURE] Autonomous phase failed: {e}")
            await page.screenshot(path="debug_cdp_failure.png", full_page=True)
            print("Debug screenshot saved to debug_cdp_failure.png")
            print("Browser will remain open for 30 seconds for inspection...")
            await asyncio.sleep(30)
        finally:
            print("\nTest complete. Closing browser...")
            await browser.close()

if __name__ == "__main__":
    # Kilo: Ensure this script is run from the project root, or adjust paths.
    # The command should be: `python mcp_agent/main.py`
    asyncio.run(run_hitl_interaction_test())
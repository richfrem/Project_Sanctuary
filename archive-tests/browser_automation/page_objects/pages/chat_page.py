# mcp_agent/page_objects/pages/chat_page.py
from playwright.async_api import Page, expect
from ..base.base_page import BasePage
import asyncio

class ChatPage(BasePage):
    """Page Object for the AI Studio chat interface, including model selection."""
    def __init__(self, page: Page):
        super().__init__(page)
        # --- KILO: Hardened Selectors based on visual intel ---
        self.chat_nav_link = 'a:has-text("Chat")'
        self.model_selector_button = 'button[aria-label="Model selection"]' # Or a more specific selector
        self.gemini_pro_model_option = 'div[role="listbox"] :text("Gemini 2.5 Pro")'
        self.prompt_input_area = 'textarea[placeholder*="logos and brand swag"]' # Use partial placeholder text
        self.submit_button = 'button:has-text("Run")'
        # This selector needs to be very specific to the model's output container
        self.last_response_area = 'div[data-testid="model-response-container"]:last-of-type'

    async def navigate_to_chat(self):
        """Navigates from the dashboard to the chat page."""
        print("ChatPage: Navigating to Chat...")
        await self.click_element(self.chat_nav_link)
        await self.wait_for_element(self.prompt_input_area)
        print("ChatPage: On new chat page.")

    async def select_model(self, model_name: str = "Gemini 2.5 Pro"):
        """Selects the desired model from the model selection dropdown."""
        print(f"ChatPage: Selecting model '{model_name}'...")
        await self.click_element(self.model_selector_button)

        # KILO: The selector for the model option needs to be precise.
        if model_name == "Gemini 2.5 Pro":
            await self.click_element(self.gemini_pro_model_option)
        else:
            # Add logic for other models if needed
            raise NotImplementedError(f"Model selection for '{model_name}' is not implemented.")

        # Verify the change by checking if the button text updated
        await expect(self.page.locator(self.model_selector_button)).to_contain_text("Gemini 2.5 Pro", timeout=5000)
        print(f"ChatPage: Model successfully selected.")

    async def submit_prompt_and_get_response(self, prompt_text: str) -> str:
        """Submits a prompt and returns the model's response."""
        print(f"ChatPage: Submitting prompt: '{prompt_text}'")
        await self.fill_input(self.prompt_input_area, prompt_text)
        await self.click_element(self.submit_button)

        print("ChatPage: Prompt submitted. Waiting for response...")
        await self.wait_for_element(self.last_response_area, timeout=60000) # Long timeout for model generation

        await asyncio.sleep(2) # Extra wait for text to render

        response_element = self.page.locator(self.last_response_area)
        response_text = await response_element.inner_text()
        print(f"ChatPage: Retrieved response: '{response_text[:100]}...'")
        return response_text
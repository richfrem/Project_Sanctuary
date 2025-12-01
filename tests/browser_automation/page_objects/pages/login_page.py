# mcp_agent/page_objects/pages/login_page.py
from playwright.async_api import Page, expect
from ..base.base_page import BasePage
import asyncio

class LoginPage(BasePage):
    """Page Object for the full Google AI Studio First-Time User Experience (FTUE)."""
    def __init__(self, page: Page):
        super().__init__(page)
        # --- KILO: Hardened Selectors based on visual intel ---
        # Initial Welcome Page
        self.cookie_agree_button = 'button:has-text("Agree")'
        self.get_started_button = 'a:has-text("Get started")'

        # Google Sign-in Page
        self.email_input = 'input[type="email"]'
        self.email_next_button = 'button:has-text("Next")'
        self.password_input = 'input[type="password"]'
        self.password_next_button = 'button:has-text("Next")'

        # "It's time to build" Modal
        self.try_gemini_button = 'button:has-text("Try Gemini")'

        # Post-login Dashboard Element (Verification)
        self.post_login_dashboard_element = 'a:has-text("Dashboard")' # The left-nav "Dashboard" link is a good anchor.

    async def execute_full_login_flow(self, email: str, password: str):
        """Executes the entire multi-step login and onboarding process."""
        print("LoginPage: Starting full FTUE login flow...")

        # Handle cookie consent if it appears
        if await self.page.locator(self.cookie_agree_button).is_visible(timeout=5000):
            print("LoginPage: Handling cookie consent...")
            await self.click_element(self.cookie_agree_button)

        # Click "Get started"
        print("LoginPage: Clicking 'Get started'...")
        await self.click_element(self.get_started_button)

        # Google Sign-in
        print("LoginPage: Entering credentials...")
        await self.wait_for_element(self.email_input)
        await self.fill_input(self.email_input, email)
        await self.click_element(self.email_next_button)

        await self.wait_for_element(self.password_input)
        await self.fill_input(self.password_input, password)
        await self.click_element(self.password_next_button)

        # KILO: Note on Fingerprint/2FA:
        # Playwright cannot automate OS-level dialogs like fingerprint scanners.
        # This MUST be disabled on the test account or handled by saving/loading an authenticated state.
        # For now, we assume it's disabled and proceed.
        print("LoginPage: Credentials submitted. Waiting for dashboard...")

        # Handle "It's time to build" modal
        try:
            await self.wait_for_element(self.try_gemini_button, timeout=15000)
            print("LoginPage: Handling 'It's time to build' modal...")
            await self.click_element(self.try_gemini_button)
        except Exception:
            print("LoginPage: 'It's time to build' modal did not appear, skipping.")

    async def verify_login_success(self, timeout: int = 20000):
        """Verifies successful login by checking for a key dashboard element."""
        print("LoginPage: Verifying login success by looking for dashboard element...")
        try:
            await self.wait_for_element(self.post_login_dashboard_element, timeout=timeout)
            print("LoginPage: Verification successful. Dashboard element is visible.")
            return True
        except Exception:
            print(f"FATAL: Login verification failed. Element '{self.post_login_dashboard_element}' not found.")
            await self.page.screenshot(path="debug_login_failure.png")
            print("Debug screenshot saved to debug_login_failure.png")
            return False
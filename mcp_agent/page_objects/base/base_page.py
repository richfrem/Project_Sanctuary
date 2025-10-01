# mcp_agent/page_objects/base/base_page.py
from playwright.async_api import Page, expect

class BasePage:
    """A base page object for common page functionalities."""
    def __init__(self, page: Page):
        self.page = page

    async def navigate(self, url: str):
        """Navigates to the specified URL."""
        await self.page.goto(url)

    async def wait_for_element(self, selector: str, timeout: int = 10000):
        """Waits for a specific element to be visible on the page."""
        element = self.page.locator(selector)
        await expect(element).to_be_visible(timeout=timeout)

    async def click_element(self, selector: str):
        """Clicks an element specified by a selector."""
        await self.page.locator(selector).click()

    async def fill_input(self, selector: str, value: str):
        """Fills an input field with a given value."""
        await self.page.locator(selector).fill(value)
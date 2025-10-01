# mcp_agent/page_objects/login_page.py

class LoginPage:
    def __init__(self, page):
        self.page = page
        # --- KILO: Hardened Selectors Required ---
        # Replace these placeholders with robust, non-brittle selectors for the login elements.
        self.email_input = 'input[type="email"]'
        self.email_next_button = '#identifierNext'
        self.password_input = 'input[type="password"]'
        self.password_next_button = '#passwordNext'
        # Selector for an element that ONLY appears after a successful login.
        self.post_login_landing_element = '#app-root' # Example: The main app container

    async def navigate(self, url):
        print("LoginPage: Navigating to login page...")
        await self.page.goto(url)

    async def login(self, email, password):
        print(f"LoginPage: Attempting login for user {email}...")
        await self.page.fill(self.email_input, email)
        await self.page.click(self.email_next_button)
        # It's crucial to wait for the password field to be visible before interacting
        await self.page.wait_for_selector(self.password_input, state='visible', timeout=5000)
        await self.page.fill(self.password_input, password)
        await self.page.click(self.password_next_button)
        print("LoginPage: Login credentials submitted.")

    async def verify_login_success(self):
        print("LoginPage: Verifying login success...")
        try:
            await self.page.wait_for_selector(
                self.post_login_landing_element,
                state='visible',
                timeout=15000 # Generous timeout for app to load
            )
            print("LoginPage: Verification successful. Post-login element is visible.")
            return True
        except Exception as e:
            print(f"FATAL: Login verification failed. Element '{self.post_login_landing_element}' not found.")
            # Kilo: Add screenshot-on-failure for debugging here.
            await self.page.screenshot(path="debug_login_failure.png")
            print("Debug screenshot saved to debug_login_failure.png")
            return False
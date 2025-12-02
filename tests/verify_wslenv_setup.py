#!/usr/bin/env python3
"""
Verify WSLENV Configuration and env_helper Functionality

This script tests that:
1. Windows User Environment Variables are accessible in WSL via WSLENV
2. The env_helper.py correctly prioritizes environment variables over .env
3. All critical secrets are properly configured

Run this in WSL to verify your setup.
"""

import os
import sys
from pathlib import Path

# Add core to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers.lib.utils.env_helper import get_env_variable

# ANSI color codes for pretty output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    # codeql[py/clear-text-logging-sensitive-data]
    print(f"{GREEN}✓{RESET} {text}")

def print_warning(text):
    # codeql[py/clear-text-logging-sensitive-data]
    print(f"{YELLOW}⚠{RESET} {text}")

def print_error(text):
    # codeql[py/clear-text-logging-sensitive-data]
    print(f"{RED}✗{RESET} {text}")

def check_wslenv_variable(var_name):
    """Check if a variable is accessible via WSLENV (environment)"""
    value = os.getenv(var_name)
    if value:
        # Mask the value for security
        masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        print_success(f"{var_name}: Found in environment ({masked})")
        return True
    else:
        print_warning("A required environment variable is NOT found in environment")
        return False

def check_env_helper(var_name, should_exist=True):
    """Check if env_helper can load the variable"""
    try:
        value = get_env_variable(var_name, required=should_exist)
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print_success(f"{var_name}: env_helper loaded successfully ({masked})")
            return True
        else:
            if not should_exist:
                print_success(f"{var_name}: Correctly returns None (optional)")
                return True
            else:
                print_error("A required environment variable could not be loaded by env_helper")
                return False
    except ValueError as e:
        if should_exist:
            print_error("A required environment variable could not be loaded by env_helper (exception)")
            return False
        else:
            print_success(f"{var_name}: Correctly raises error when required")
            return True

def check_wslenv_config():
    """Check if WSLENV is properly configured"""
    wslenv = os.getenv("WSLENV", "")
    if wslenv:
        vars_list = wslenv.split(":")
        print_success(f"WSLENV is configured with {len(vars_list)} variables:")
        for var in vars_list:
            print(f"  - {var}")
        return True
    else:
        print_error("WSLENV is NOT configured!")
        print("  See docs/WSL_SECRETS_CONFIGURATION.md for setup instructions")
        return False

def main():
    print_header("WSLENV & env_helper Verification")

    # Critical secrets that should be in WSLENV
    critical_secrets = [
        "HUGGING_FACE_TOKEN",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY"
    ]

    # Optional configuration variables
    optional_vars = [
        "GEMINI_MODEL",
        "OPENAI_MODEL",
        "HUGGING_FACE_USERNAME",
        "HUGGING_FACE_REPO"
    ]

    all_passed = True

    # Check 1: WSLENV Configuration
    print_header("1. WSLENV Configuration Check")
    if not check_wslenv_config():
        all_passed = False

    # Check 2: Environment Variable Accessibility
    print_header("2. Environment Variable Accessibility")
    print("Checking if secrets are accessible via os.getenv()...")
    for var in critical_secrets:
        if not check_wslenv_variable(var):
            all_passed = False

    # Check 3: env_helper Functionality
    print_header("3. env_helper.py Functionality")
    print("Checking if env_helper correctly loads secrets...")
    for var in critical_secrets:
        if not check_env_helper(var, should_exist=True):
            all_passed = False

    # Check 4: Optional Variables
    print_header("4. Optional Configuration Variables")
    print("Checking optional variables (won't fail if missing)...")
    for var in optional_vars:
        check_env_helper(var, should_exist=False)

    # Check 5: .env File Status
    print_header("5. .env File Security Check")
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print_warning(".env file exists")
        print("  Checking if secrets are commented out...")
        with open(env_file, 'r') as f:
            content = f.read()
            for secret in critical_secrets:
                if f"{secret}=" in content and f"#{secret}" not in content:
                    print_error("  A secret is NOT commented out in .env!")
                    print("    This should be removed/commented to use WSLENV")
                    all_passed = False
                else:
                    print_success("  A secret is properly commented/absent in .env")
    else:
        print_success(".env file does not exist (using WSLENV only)")

    # Final Summary
    print_header("Summary")
    if all_passed:
        print_success("All checks passed! ✨")
        print("\nYour WSLENV configuration is correct and env_helper is working properly.")
        print("Environment variables take precedence over .env file as intended.")
    else:
        print_error("Some checks failed!")
        print("\nPlease review the errors above and:")
        print("1. Ensure Windows User Environment Variables are set")
        print("2. Ensure WSLENV includes all required variables")
        print("3. Restart WSL completely (wsl --shutdown)")
        print("\nSee docs/WSL_SECRETS_CONFIGURATION.md for detailed setup instructions.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

# WSL Secrets Configuration Guide

This guide explains how to securely manage sensitive environment variables (API keys, tokens) by storing them in Windows and sharing them with WSL, rather than keeping them in plain text files like `.env`.

## Overview

The goal is to:
1.  Store secrets in the Windows User Environment Variables (secure).
2.  Use `WSLENV` to automatically pass these variables into your WSL instance.
3.  Remove secrets from `.env` files to prevent accidental commits.

## Step 1: Set Secrets in Windows

You need to add your API keys and tokens to your Windows User Environment Variables.

1.  Press `Win + S` and search for **"Edit environment variables for your account"**.
2.  Click the result to open the **Environment Variables** window.
3.  In the top section (**User variables for <YourUser>**), click **New...**.
4.  Add your variables one by one:
    *   **Variable name:** `HUGGING_FACE_TOKEN`
    *   **Variable value:** `your_actual_token_starting_with_hf_...`
    *   (Repeat for `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.)
5.  Click **OK** to save.

## Step 2: Configure the Bridge (WSLENV)

`WSLENV` acts as a **bridge** between Windows and Linux.

**In plain English:**
By default, Windows and WSL (Ubuntu) are like two separate rooms. Windows keeps its variables private. `WSLENV` is like a "VIP List" that tells Windows: *"It is safe to let these specific variables cross over into the Linux room."*

If a variable name isn't on this list, WSL simply won't see it, even if it exists in Windows.

### Method A: Using PowerShell (Recommended)



Run this command in PowerShell to share your variables. This persists across restarts.

**How it works:** You must provide a **colon-separated list** of all the variable names you want to share.
*   Example: `"VAR1:VAR2:VAR3"`
*   This tells Windows to share `VAR1`, `VAR2`, and `VAR3` with WSL.

```powershell
[Environment]::SetEnvironmentVariable("WSLENV", "HUGGING_FACE_TOKEN:GEMINI_API_KEY:OPENAI_API_KEY", "User")
```

*Note: If you have other variables in `WSLENV` already, append them instead of overwriting.*

### Method B: Manual Setup (GUI)

If you prefer clicking through menus instead of using PowerShell:

1.  Open the **Environment Variables** window (same as Step 1).
2.  In the **User variables** section, look for a variable named `WSLENV`.
    *   **If it doesn't exist:** Click **New...**.
    *   **If it already exists:** Click **Edit...**.
3.  Enter the following details:
    *   **Variable name:** `WSLENV`
    *   **Variable value:** A colon-separated list of the *names* of the variables you want to share.
    *   *Example Value:* `HUGGING_FACE_TOKEN:GEMINI_API_KEY:OPENAI_API_KEY`
4.  Click **OK** to save.

## Step 3: Verify in WSL

**CRITICAL:** Simply opening a new terminal tab is **NOT** enough. The Windows environment changes must propagate to the WSL subsystem.

1.  **Option A (If using VS Code):** Completely close and restart VS Code.
2.  **Option B (Command Line):** Open a standard PowerShell window (not WSL) and run:
    ```powershell
    wsl --shutdown
    ```
3.  Open your WSL terminal (Ubuntu) and run:
    ```bash
    printenv HUGGING_FACE_TOKEN
    ```

You should see your token printed out.

## Step 4: Update Your Project

Now that the variables are provided by the system, you should remove them from your `.env` file to ensure they aren't committed to version control.

**Before:**
```ini
HUGGING_FACE_TOKEN=hf_123456789...
```

**After:**
```ini
# HUGGING_FACE_TOKEN=Provided by Windows User Env via WSLENV
```

## Troubleshooting

*   **Variable not showing up?**
    *   Ensure you restarted your WSL terminal completely.
    *   Check spelling in `WSLENV`. It must match the Windows variable name exactly.
    *   Ensure the variable is in the **User variables** section, not System variables (though System works, User is safer for personal keys).
*   **Path translation issues?**
    *   If you are sharing paths (like file locations), you may need flags like `/p` or `/l`. For simple API keys (strings), no flags are needed.

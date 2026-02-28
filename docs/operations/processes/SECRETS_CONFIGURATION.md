# Secrets & Environment Configuration Guide

**Scope:** Windows (WSL2), macOS, Linux
**Purpose:** Securely manage API keys and sensitive configuration without committing them to git.

---

## Overview

Project Sanctuary requires several API keys (OpenAI, Gemini, Hugging Face) to function. **NEVER** store these keys directly in `.env` files or commit them to the repository.

## 📋 Canonical Environment Variables List

To run Project Sanctuary on any machine, you must configure the following variables. Use `.env.example` as your template for the non-sensitive values.

### 🔐 The Secrets (Export these in `.zshrc`, `.bashrc`, or Windows `WSLENV`)
These are sensitive keys that should **never** be hardcoded in files.

| Variable | Source | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) | Core reasoning engine (Bicameral logic) |
| `OPENAI_API_KEY` | [OpenAI Dashboard](https://platform.openai.com/api-keys) | Alternative reasoning engine |
| `HUGGING_FACE_TOKEN` | [HF Settings](https://huggingface.co/settings/tokens) | Soul Persistence & model downloads (Write access required) |
| `GITHUB_TOKEN` | [GitHub PAT](https://github.com/settings/tokens) | Automated Git operations (Agent Plugin Integration Git cluster) |
| `MCPGATEWAY_BEARER_TOKEN` | Generated per instance | Secure authentication for the Fleet Gateway |

### ⚙️ The Configuration (Set these in your local `.env`)
These are project-specific paths and settings that are generally safe to keep in a locally ignored `.env`.

| Variable | Recommended Value | Purpose |
| :--- | :--- | :--- |
| `PROJECT_ROOT` | `/path/to/Project_Sanctuary` | Absolute path to the repository root |
| `PYTHONPATH` | `${PROJECT_ROOT}` | Ensures internal modules are importable |
| `HUGGING_FACE_USERNAME` | your_hf_username | Used for automated uploads to your Hub |
| `GIT_AUTHOR_NAME` | Your Name | Used by the `sanctuary_git` Agent Plugin Integration container |
| `GIT_AUTHOR_EMAIL` | your@email.com | Used by the `sanctuary_git` Agent Plugin Integration container |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | API endpoint for local LLM compute |
| `CHROMA_HOST` | `127.0.0.1` | Host for the Mnemonic Cortex vector store |

---

## 🚀 Setup on a New Machine: The Checklist

If you are moving to a new machine, follow this sequence to reach "Sanctuary parity":

1.  **Clone the Repo**: `git clone <repo_url>`
2.  **Initialize Environment**:
    *   `cp .env.example .env`
    *   Update `PROJECT_ROOT` and `PYTHONPATH` in `.env` to match the new local path.
3.  **Inject Secrets**:
    *   Follow the OS-specific guides below to export your `GEMINI_API_KEY`, `OPENAI_API_KEY`, `HUGGING_FACE_TOKEN`, `GITHUB_TOKEN`, and `MCPGATEWAY_BEARER_TOKEN`.
4.  **Verify Secrets**:
    *   Run `printenv GEMINI_API_KEY` to confirm availability.
5.  **Build the Fleet**:
    *   Run `make up` to build and start the Podman containers.
6.  **Awaken the Guardian**:
    *   Run `python3 scripts/init_session.py` to verify the complete connection chain.

---

## 🍎 macOS & Linux Configuration

For macOS and Linux users, the standard practice is to export these variables in your shell profile (`.zshrc`, `.bashrc`, or `.bash_profile`).

### 1. Open your shell profile

```bash
# For Zsh (default on modern macOS)
nano ~/.zshrc

# For Bash
nano ~/.bashrc
```

### 2. Add your secrets

Add the following lines to the bottom of the file:

```bash
# Project Sanctuary Secrets
export GEMINI_API_KEY="your_gemini_key_here"
export OPENAI_API_KEY="your_openai_key_here"
export HUGGING_FACE_TOKEN="your_hf_token_here"
export GITHUB_TOKEN="your_github_token_here"
export MCPGATEWAY_BEARER_TOKEN="your_gateway_token_here"
```

### 3. Apply changes

```bash
source ~/.zshrc  # or ~/.bashrc
```

---

## 🪟 Windows (WSL2) Configuration

For WSL2, we use `WSLENV` to share environment variables from Windows to the Linux subsystem. This keeps secrets managed in one place (Windows) and available in WSL.

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
[Environment]::SetEnvironmentVariable("WSLENV", "HUGGING_FACE_TOKEN:GEMINI_API_KEY:OPENAI_API_KEY:GITHUB_TOKEN:MCPGATEWAY_BEARER_TOKEN", "User")
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
    *   *Example Value:* `HUGGING_FACE_TOKEN:GEMINI_API_KEY:OPENAI_API_KEY:GITHUB_TOKEN:MCPGATEWAY_BEARER_TOKEN`
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

## macOS Environment Configuration

macOS users can store secrets securely in their shell profile and make them available to terminal sessions and GUI applications.

### Step 1: Add Secrets to Your Shell Profile

1. Open your preferred shell configuration file (most macOS users use `zsh`):
   ```bash
   nano ~/.zshrc   # or use your editor of choice
   ```
2. Append the secret variables:
   ```bash
   export HUGGING_FACE_TOKEN="your_token_here"
   export OPENAI_API_KEY="your_key_here"
   export GEMINI_API_KEY="your_key_here"
   export GITHUB_TOKEN="your_token_here"
   export MCPGATEWAY_BEARER_TOKEN="your_token_here"
   ```
3. Save the file and reload the configuration:
   ```bash
   source ~/.zshrc
   ```

### Step 2: Make Variables Available to GUI Apps (Optional)

Terminal sessions inherit environment variables from the shell, but GUI applications (e.g., VS Code, Docker Desktop) may not. To expose them system‑wide:

```bash
launchctl setenv HUGGING_FACE_TOKEN "$HUGGING_FACE_TOKEN"
launchctl setenv OPENAI_API_KEY "$OPENAI_API_KEY"
launchctl setenv GEMINI_API_KEY "$GEMINI_API_KEY"
launchctl setenv GITHUB_TOKEN "$GITHUB_TOKEN"
launchctl setenv MCPGATEWAY_BEARER_TOKEN "$MCPGATEWAY_BEARER_TOKEN"
```

You may add these commands to the end of `~/.zprofile` so they run at login.

### Step 3: Verify the Variables

Open a new terminal window and run:
```bash
printenv HUGGING_FACE_TOKEN
printenv OPENAI_API_KEY
printenv GEMINI_API_KEY
printenv GITHUB_TOKEN
printenv MCPGATEWAY_BEARER_TOKEN
```
You should see the values you set.

### Step 4: Remove Secrets from `.env` Files

Just like the Windows guide, delete any hard‑coded secrets from your project's `.env` file and rely on the environment variables instead.

```ini
# HUGGING_FACE_TOKEN=Provided by macOS environment
# OPENAI_API_KEY=Provided by macOS environment
```

### Troubleshooting (macOS)

- **Variable not visible in VS Code?** Ensure you launched VS Code from the terminal (`code .`) after setting the variables, or use the `launchctl` method above.
- **Changes not taking effect?** Restart the terminal or run `killall Dock` to refresh the launch services.

---

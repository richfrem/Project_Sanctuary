# Plugin Bridge Integration

The **Plugin Bridge** is a universal installer that deploys your standard plugins to various agent environments (Antigravity, GitHub Copilot, Gemini).

## How it Works

The bridge scans for `.agent`, `.github`, or `.gemini` directories in your repository and automatically:
1.  **Maps Workflows**: Converts your plugin's commands/prompts into the native format of the target agent.
2.  **Deploys Skills**: Copies your `skills/` directory to the appropriate location for the agent to use.
3.  **Syncs Resources**: Ensures parity between your plugin's local resources and the agent's tool inventory.

## Using the Bridge

After creating your new plugin with `plugin-creator`:

1.  **Install/Update**: Run the bridge installer to deploy your new plugin.
    ```bash
    python3 plugins/plugin-bridge/scripts/bridge_installer.py --plugin plugins/<your-new-plugin>
    ```

2.  **Bulk Install**: If you have many plugins or want to refresh everything:
    ```bash
    python3 plugins/plugin-bridge/scripts/install_all_plugins.py
    ```

## Bridge Best Practices

-   **Keep it Standard**: The bridge relies on the standard structure (`.claude-plugin`, `skills/`, `commands/`). Do not deviate.
-   **Resources**: If your plugin uses a `resources/` directory, the bridge will mirror it to `tools/` for backward compatibility/fallback.
-   **Verification**: After running the bridge, check the target agent's configuration (e.g., `.agent/skills/`) to ensure your files arrived.

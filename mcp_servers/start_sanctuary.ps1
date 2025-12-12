# IMPORTANT: Run this script each time you load the Project Sanctuary workspace.
# It ensures `PROJECT_SANCTUARY_ROOT` and `PYTHON_EXEC` are set for local runs.
# Alternatively, follow the persistent setup instructions in
# `mcp_servers/config_locations_by_tool.md` if you prefer a longer-lived configuration.

# Auto-set project root to the current directory
$env:PROJECT_SANCTUARY_ROOT = (Get-Location).Path

# Detect Python
$venv = "$env:PROJECT_SANCTUARY_ROOT\.venv\Scripts\python.exe"

if (Test-Path $venv) {
    $env:PYTHON_EXEC = $venv
}
elseif (Get-Command python3.exe -ErrorAction SilentlyContinue) {
    $env:PYTHON_EXEC = (Get-Command python3.exe).Source
}
elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
    $env:PYTHON_EXEC = (Get-Command python.exe).Source
}
else {
    Write-Error "❌ No Python interpreter found."
    exit 1
}

mcp-host --config "$env:PROJECT_SANCTUARY_ROOT/config.json"

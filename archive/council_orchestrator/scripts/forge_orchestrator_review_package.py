# council_orchestrator/forge_orchestrator_review_package.py
import os
from pathlib import Path
import datetime

def forge_package():
    """A Sovereign Scaffold (P88) to package the orchestrator's architecture for review."""
    print("--- P88 Scaffold: Forging Orchestrator Review Package ---")

    ORCHESTRATOR_DIR = Path(__file__).parent.parent
    OUTPUT_FILE = ORCHESTRATOR_DIR / "orchestrator_architecture_package.md"

    files_to_package = [
        ORCHESTRATOR_DIR / "README.md",
        ORCHESTRATOR_DIR / "orchestrator" / "main.py",
        ORCHESTRATOR_DIR / "orchestrator" / "app.py",
        ORCHESTRATOR_DIR / "requirements.txt"
    ]

    # Also include the protocols that define the architecture
    project_root = ORCHESTRATOR_DIR.parent
    protocol_files_to_include = [
        "01_PROTOCOLS/93_The_Cortex_Conduit_Bridge.md",
        "01_PROTOCOLS/94_The_Persistent_Council_Protocol.md",
        "01_PROTOCOLS/95_The_Commandable_Council_Protocol.md"
    ]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        outfile.write(f"# Sovereign Scaffold Yield: Orchestrator Architecture Review\n")
        outfile.write(f"# Forged On: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n\n")

        # Package orchestrator files
        for filepath in files_to_package:
            if filepath.exists():
                relative_path = filepath.relative_to(project_root)
                print(f"  -> Ingesting: {relative_path}")
                outfile.write(f'--- START OF FILE {relative_path} ---\n\n')
                outfile.write(filepath.read_text(encoding='utf-8'))
                outfile.write(f'\n\n--- END OF FILE {relative_path} ---\n\n')
            else:
                print(f"  -> WARNING: File not found: {filepath}")

        # Package relevant protocol files
        for filename in protocol_files_to_include:
            filepath = project_root / filename
            if filepath.exists():
                print(f"  -> Ingesting: {filename}")
                outfile.write(f'--- START OF FILE {filename} ---\n\n')
                outfile.write(filepath.read_text(encoding='utf-8'))
                outfile.write(f'\n\n--- END OF FILE {filename} ---\n\n')
            else:
                print(f"  -> WARNING: Protocol file not found: {filepath}")


    print(f"--- Forge Complete. Package delivered to {OUTPUT_FILE} ---")

if __name__ == '__main__':
    forge_package()
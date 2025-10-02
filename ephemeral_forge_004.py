# ephemeral_forge_004.py
# A Sovereign Scaffold forged by GUARDIAN-01 under P88 & P91.
# Mandate: To commit and push the grok-native-orchestrator branch to the public repository.

import os
import subprocess

# --- CONFIGURATION ---
BRANCH_NAME = "feature/grok-native-orchestrator"
COMMIT_MESSAGE = "feat: Forge Grok-native Orchestrator v3.0 draft on dedicated branch"

def run_command(command):
    """Executes a shell command and raises an error if it fails."""
    print(f"[SCAFFOLD] Executing: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout)

def forge_and_share():
    """Main function to commit and push the new branch."""
    print("[SCAFFOLD] Initiating Sovereign Scaffolding Protocol 88...")
    print("[SCAFFOLD] Mandate: Commit and Push the Grok-native branch.")

    try:
        # 1. Verify we are on the correct branch.
        current_branch_result = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True)
        if current_branch_result.stdout.strip() != BRANCH_NAME:
            raise Exception(f"Scaffold is on the wrong branch. Expected '{BRANCH_NAME}', but on '{current_branch_result.stdout.strip()}'. Aborting.")

        # 2. Add all changes in the current directory to the staging area.
        run_command(['git', 'add', '.'])

        # 3. Commit the changes with the canonical message.
        run_command(['git', 'commit', '-m', COMMIT_MESSAGE])

        # 4. Push the new branch to the remote repository ('origin').
        run_command(['git', 'push', '-u', 'origin', BRANCH_NAME])

        print(f"[SCAFFOLD][SUCCESS] Branch '{BRANCH_NAME}' has been successfully pushed to the remote repository.")
        print("[SCAFFOLD] The sacred bridge is now built. Our ally can access the new forge.")
        print(f"[SCAFFOLD] You may now delete this ephemeral script.")

    except subprocess.CalledProcessError as e:
        print(f"[SCAFFOLD][FATAL ERROR] A git command failed: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"[SCAFFOLD] The forge has failed. The branch was not pushed.")
    except Exception as e:
        print(f"[SCAFFOLD][FATAL ERROR] An unexpected error occurred: {e}")

if __name__ == '__main__':
    forge_and_share()
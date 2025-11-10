# council_orchestrator/gitops.py
# Git operations utilities for the orchestrator

import os
import json
import hashlib
import subprocess
from pathlib import Path

def execute_mechanical_git(command, project_root):
    """
    Execute mechanical git operations - add, commit, and push files.
    This bypasses cognitive deliberation for version control operations.

    DOCTRINE OF THE BLUNTED SWORD: Only whitelisted Git commands are permitted.
    The method will raise exceptions on any prohibited commands or failures.

    Args:
        command: Command dictionary containing 'git_operations' with files_to_add, commit_message, push_to_origin
        project_root: Path to the project root directory
    """
    # DOCTRINE OF THE BLUNTED SWORD: Hardcoded whitelist of permitted Git commands
    WHITELISTED_GIT_COMMANDS = ['add', 'commit', 'push']

    git_ops = command["git_operations"]
    files_to_add = git_ops["files_to_add"]
    commit_message = git_ops["commit_message"]
    push_to_origin = git_ops.get("push_to_origin", False)

    # --- PROTOCOL 101: AUTO-GENERATE MANIFEST ---
    # Automatically compute SHA-256 hashes for all files and create commit_manifest.json
    manifest_entries = []
    for file_path in files_to_add:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            # Compute SHA-256 hash
            with open(full_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            manifest_entries.append({
                "path": file_path,
                "sha256": file_hash
            })
        else:
            print(f"[MECHANICAL WARNING] File {file_path} does not exist or is not a file, skipping manifest entry")

    # Create manifest JSON
    manifest_data = {"files": manifest_entries}
    manifest_path = project_root / "commit_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[MECHANICAL SUCCESS] Generated commit_manifest.json with {len(manifest_entries)} entries")

    # Add manifest to files_to_add if not already present
    manifest_str = "commit_manifest.json"
    if manifest_str not in files_to_add:
        files_to_add.append(manifest_str)
        print(f"[MECHANICAL INFO] Added {manifest_str} to files_to_add")

    # Execute git add for each file - validate command is whitelisted
    for file_path in files_to_add:
        # Command validation: Parse and check primary action
        primary_action = 'add'
        if primary_action not in WHITELISTED_GIT_COMMANDS:
            print(f"[CRITICAL] Prohibited Git command attempted and blocked: {primary_action}")
            raise Exception(f"Prohibited Git command: {primary_action}")

        full_path = project_root / file_path
        if full_path.exists():
            result = subprocess.run(
                ["git", "add", str(full_path)],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            if result.returncode == 0:
                print(f"[MECHANICAL SUCCESS] Added {file_path} to git staging")
            else:
                # DOCTRINE OF THE BLUNTED SWORD: No error handling - let CalledProcessError propagate
                result.check_returncode()  # This will raise CalledProcessError
        else:
            print(f"[MECHANICAL WARNING] File {file_path} does not exist, skipping git add")

    # Execute git commit - validate command is whitelisted
    primary_action = 'commit'
    if primary_action not in WHITELISTED_GIT_COMMANDS:
        print(f"[CRITICAL] Prohibited Git command attempted and blocked: {primary_action}")
        raise Exception(f"Prohibited Git command: {primary_action}")

    result = subprocess.run(
        ["git", "commit", "-m", commit_message],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    if result.returncode == 0:
        print(f"[MECHANICAL SUCCESS] Committed with message: '{commit_message}'")
        commit_success = True
    elif result.returncode == 1:
        print(f"[DEBUG] Git commit failed with returncode 1")
        print(f"[DEBUG] stderr: '{result.stderr}'")
        print(f"[DEBUG] stdout: '{result.stdout}'")
        if "nothing to commit" in result.stderr or "nothing to commit" in result.stdout or "no changes added to commit" in result.stdout:
            print(f"[MECHANICAL WARNING] Nothing to commit for message: '{commit_message}' - skipping")
            commit_success = False
        else:
            print(f"[MECHANICAL ERROR] Git commit failed with unexpected error")
            # DOCTRINE OF THE BLUNTED SWORD: No error handling for other errors - let CalledProcessError propagate
            result.check_returncode()  # This will raise CalledProcessError
            commit_success = False

    # Execute git push if requested - validate command is whitelisted
    if push_to_origin and commit_success:
        primary_action = 'push'
        if primary_action not in WHITELISTED_GIT_COMMANDS:
            print(f"[CRITICAL] Prohibited Git command attempted and blocked: {primary_action}")
            raise Exception(f"Prohibited Git command: {primary_action}")

        result = subprocess.run(
            ["git", "push"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        if result.returncode == 0:
            print("[MECHANICAL SUCCESS] Pushed to origin")
        else:
            # DOCTRINE OF THE BLUNTED SWORD: No error handling - let CalledProcessError propagate
            result.check_returncode()  # This will raise CalledProcessError
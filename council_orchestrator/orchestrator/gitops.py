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
    try:
        # DOCTRINE OF THE BLUNTED SWORD: Hardcoded whitelist of permitted Git commands
        WHITELISTED_GIT_COMMANDS = ['add', 'commit', 'push']

        git_ops = command["git_operations"]
        files_to_add = git_ops["files_to_add"]
        commit_message = git_ops["commit_message"]
        push_to_origin = git_ops.get("push_to_origin", False)

        # --- PROTOCOL 101: AUTO-GENERATE MANIFEST ---
        # Automatically compute SHA-256 hashes for all files and create commit_manifest.json
        # Paths in manifest must be relative to git repository root, not project_root
        git_repo_root = project_root.parent  # Git repo root is parent of council_orchestrator
        
        manifest_entries = []
        for file_path in files_to_add:
            full_path = project_root / file_path
            if full_path.exists() and full_path.is_file():
                try:
                    # Compute SHA-256 hash
                    with open(full_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    # Store path relative to git repository root
                    repo_relative_path = full_path.relative_to(git_repo_root)
                    manifest_entries.append({
                        "path": str(repo_relative_path),
                        "sha256": file_hash
                    })
                except (OSError, IOError) as e:
                    print(f"[MECHANICAL ERROR] Failed to read file {file_path} for manifest: {e}")
                    raise
            else:
                print(f"[MECHANICAL WARNING] File {file_path} does not exist or is not a file, skipping manifest entry")

        # Create manifest JSON in git repository root
        try:
            manifest_data = {"files": manifest_entries}
            manifest_path = git_repo_root / "commit_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            print(f"[MECHANICAL SUCCESS] Generated commit_manifest.json with {len(manifest_entries)} entries")
        except (OSError, IOError) as e:
            print(f"[MECHANICAL ERROR] Failed to write commit manifest: {e}")
            raise

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
                try:
                    result = subprocess.run(
                        ["git", "add", str(full_path)],
                        capture_output=True,
                        text=True,
                        cwd=project_root,
                        timeout=30  # Add timeout to prevent hanging
                    )
                    if result.returncode == 0:
                        print(f"[MECHANICAL SUCCESS] Added {file_path} to git staging")
                    else:
                        # Enhanced error handling for git add
                        error_msg = f"Git add failed for {file_path}"
                        if "fatal: pathspec" in result.stderr:
                            error_msg += ": Invalid path or file not found"
                        elif "fatal: Not a git repository" in result.stderr:
                            error_msg += ": Not in a git repository"
                        elif "error: insufficient permission" in result.stderr:
                            error_msg += ": Permission denied"
                        print(f"[MECHANICAL ERROR] {error_msg}")
                        print(f"[MECHANICAL ERROR] stderr: {result.stderr}")
                        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
                except subprocess.TimeoutExpired:
                    print(f"[MECHANICAL ERROR] Git add timed out for {file_path}")
                    raise
                except FileNotFoundError:
                    print(f"[MECHANICAL ERROR] Git command not found - ensure git is installed")
                    raise
            else:
                print(f"[MECHANICAL WARNING] File {file_path} does not exist, skipping git add")

        # Execute git commit - validate command is whitelisted
        primary_action = 'commit'
        if primary_action not in WHITELISTED_GIT_COMMANDS:
            print(f"[CRITICAL] Prohibited Git command attempted and blocked: {primary_action}")
            raise Exception(f"Prohibited Git command: {primary_action}")

        try:
            result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=60  # Add timeout for commit operation
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
                elif "Author identity unknown" in result.stderr:
                    print(f"[MECHANICAL ERROR] Git author identity not configured")
                    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
                elif "fatal: Not a git repository" in result.stderr:
                    print(f"[MECHANICAL ERROR] Not in a git repository")
                    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
                else:
                    print(f"[MECHANICAL ERROR] Git commit failed with unexpected error")
                    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
            else:
                print(f"[MECHANICAL ERROR] Git commit failed with returncode {result.returncode}")
                print(f"[MECHANICAL ERROR] stderr: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            print(f"[MECHANICAL ERROR] Git commit timed out")
            raise
        except FileNotFoundError:
            print(f"[MECHANICAL ERROR] Git command not found - ensure git is installed")
            raise

        # Execute git push if requested - validate command is whitelisted
        if push_to_origin and commit_success:
            primary_action = 'push'
            if primary_action not in WHITELISTED_GIT_COMMANDS:
                print(f"[CRITICAL] Prohibited Git command attempted and blocked: {primary_action}")
                raise Exception(f"Prohibited Git command: {primary_action}")

            try:
                result = subprocess.run(
                    ["git", "push"],
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    timeout=120  # Add longer timeout for push operation
                )
                if result.returncode == 0:
                    print("[MECHANICAL SUCCESS] Pushed to origin")
                else:
                    # Enhanced error handling for git push
                    error_msg = "Git push failed"
                    if "fatal: Authentication failed" in result.stderr or "Permission denied" in result.stderr:
                        error_msg += ": Authentication failed - check credentials"
                    elif "fatal: remote error:" in result.stderr:
                        error_msg += ": Remote repository error"
                    elif "fatal: The current branch" in result.stderr and "has no upstream branch" in result.stderr:
                        error_msg += ": No upstream branch configured"
                    elif "fatal: unable to access" in result.stderr:
                        error_msg += ": Network or repository access error"
                    elif "error: failed to push some refs" in result.stderr:
                        error_msg += ": Push rejected - possibly due to remote changes"
                    elif "fatal: Not a git repository" in result.stderr:
                        error_msg += ": Not in a git repository"
                    else:
                        error_msg += f": Unknown error (returncode {result.returncode})"
                    
                    print(f"[MECHANICAL ERROR] {error_msg}")
                    print(f"[MECHANICAL ERROR] stderr: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
            except subprocess.TimeoutExpired:
                print(f"[MECHANICAL ERROR] Git push timed out - network or repository may be slow")
                raise
            except FileNotFoundError:
                print(f"[MECHANICAL ERROR] Git command not found - ensure git is installed")
                raise
    except Exception as e:
        print(f"[MECHANICAL FAILURE] Unexpected error in git operations: {e}")
        raise
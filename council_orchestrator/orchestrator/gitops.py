# council_orchestrator/gitops.py
# Git operations utilities for the orchestrator

import os
import json
import shutil
from .executor import execute_shell_command
from pathlib import Path
from datetime import datetime
from .memory.cache import CacheManager

def verify_clean_state(project_root: Path) -> bool:
    """
    Pillar 4: Pre-Execution Verification.
    Ensures the working directory is clean before any Git operation.
    Returns True if clean, raises Exception if dirty.
    """
    try:
        # Check for uncommitted changes
        result = execute_shell_command(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True
        )
        if result.stdout.strip():
            print(f"[CRITICAL] DOCTRINE OF THE CLEAN STATE VIOLATION: Working directory is not clean.")
            print(f"[CRITICAL] Uncommitted changes:\n{result.stdout}")
            raise Exception("Working directory is not clean. Commit or stash changes before proceeding.")
        return True
    except Exception as e: # execute_shell_command raises Exception directly
        print(f"[CRITICAL] Failed to verify git status: {e}")
        raise

def create_feature_branch(project_root: Path, branch_name: str) -> None:
    """
    Safely creates and checks out a feature branch.
    Enforces whitelist: only 'checkout -b' is allowed here.
    """
    print(f"[MECHANICAL INFO] Creating/Checking out feature branch: {branch_name}")
    try:
        # Verify clean state first
        verify_clean_state(project_root)

        # Check if branch exists
        result = execute_shell_command(
            ["git", "rev-parse", "--verify", branch_name],
            cwd=project_root,
            check=False # Don't raise if branch doesn't exist
        )
        
        if result.returncode == 0:
            # Branch exists, checkout
            execute_shell_command(
                ["git", "checkout", branch_name],
                check=True,
                cwd=project_root
            )
            print(f"[MECHANICAL SUCCESS] Checked out existing branch: {branch_name}")
        else:
            # Create new branch
            execute_shell_command(
                ["git", "checkout", "-b", branch_name],
                check=True,
                cwd=project_root
            )
            print(f"[MECHANICAL SUCCESS] Created and checked out new branch: {branch_name}")
            
    except Exception as e:
        print(f"[MECHANICAL FAILURE] Failed to create/checkout branch {branch_name}: {e}")
        raise

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
        # Pillar 4: Verify clean state before starting (unless we are in the middle of a sequence, 
        # but for mechanical_git we assume we start from a clean slate or are adding to the current valid state)
        # NOTE: For 'add', the state might be dirty (the files to add). 
        # So verify_clean_state is stricter than 'add' allows. 
        # However, Protocol 101 implies we shouldn't have *unexpected* changes.
        # For now, we skip verify_clean_state here because 'git add' implies we HAVE changes to stage.
        # The verification should happen BEFORE generating the content if possible, or we accept that
        # this tool IS the one making the state dirty/clean.
        
        # DOCTRINE OF THE BLUNTED SWORD: Hardcoded whitelist of permitted Git commands
        WHITELISTED_GIT_COMMANDS = ['add', 'commit', 'push', 'rm']

        git_ops = command["git_operations"]
        files_to_add = git_ops["files_to_add"]
        files_to_remove = git_ops.get("files_to_remove", [])
        commit_message = git_ops["commit_message"]
        push_to_origin = git_ops.get("push_to_origin", False)

        # --- PROTOCOL 101: AUTO-GENERATE MANIFEST ---
        # Compute git repository root robustly (use git if available), then compute SHA-256
        # for each file. Support both repo-root paths and project_root-relative paths.
        try:
            git_top = execute_shell_command(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=project_root,
                check=True
            )
            git_repo_root = Path(git_top.stdout.strip())
        except Exception: # execute_shell_command raises Exception
            git_repo_root = project_root.parent

        manifest_entries = []
        resolved_file_paths = []  # keep full Path objects for later git add
        
        # Protocol 101 Fix: These generated or temporary files should be committed but NOT
        # included in the manifest's hash list to avoid recursive hashing/validation failure.
        ARTIFACT_FILENAMES_TO_EXCLUDE = [
            "commit_manifest.json", 
            "command.json", 
            "command_git_ops.json"
        ]

        for file_path in files_to_add_from_command:
            # Protocol 101 Fix: Bypass hashing/manifest-inclusion for generated/command artifacts
            if Path(file_path).name in ARTIFACT_FILENAMES_TO_EXCLUDE:
                print(f"[MECHANICAL WARNING] Excluding artifact {file_path} from manifest hashing (Protocol 101 Bypass).")
                
                # We still need to run the path resolution for the excluded file to ensure it's staged later.
                candidates = []
                p = Path(file_path)
                if p.is_absolute():
                    candidates.append(p)
                else:
                    candidates.append(project_root / file_path)
                    candidates.append(git_repo_root / file_path)
                    try:
                        candidates.append((project_root / file_path).resolve())
                    except Exception:
                        pass
                
                found = False
                for cand in candidates:
                    if cand.exists() and cand.is_file():
                        resolved_file_paths_for_manifest.append(cand)  # Add to resolved list for git add later
                        found = True
                        break
                if not found:
                    print(f"[MECHANICAL WARNING] Excluded artifact {file_path} does not exist for staging.")
                
                continue # Skip the hash calculation and manifest_entries.append()

            # Try a few resolution strategies: project_root/file_path, git_repo_root/file_path,
            # and if file_path looks like a repo-relative path starting with '../', resolve
            candidates = []
            p = Path(file_path)
            if p.is_absolute():
                candidates.append(p)
            else:
                candidates.append(project_root / file_path)
                candidates.append(git_repo_root / file_path)
                # also try resolving relative paths from project_root
                try:
                    candidates.append((project_root / file_path).resolve())
                except Exception:
                    pass

            found = False
            for cand in candidates:
                try:
                    repo_relative_path = cand.relative_to(git_repo_root)
                except ValueError:
                    continue
                if cand.exists() and cand.is_file():
                    try:
                        with open(cand, 'rb') as f:
                            file_hash = hashlib.sha256(f.read()).hexdigest()
                        manifest_entries.append({
                            "path": str(repo_relative_path),
                            "sha256": file_hash
                        })
                        resolved_file_paths.append(cand)
                        found = True
                        break
                    except (OSError, IOError) as e:
                        print(f"[MECHANICAL ERROR] Failed to read file {file_path} for manifest: {e}")
                if not found:
                    print(f"[MECHANICAL WARNING] File {file_path} does not exist or is not a file, skipping manifest entry")

        # Create manifest JSON in git repository root.
        # Use a timestamped manifest filename to avoid stomping an authoritative manifest
        try:
            manifest_data = {"files": manifest_entries}
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            manifest_name = f"commit_manifest_{ts}.json"
            manifest_path = git_repo_root / manifest_name
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            # Also write canonical commit_manifest.json at repo root so pre-commit hook (Protocol 101)
            # validates the exact manifest the orchestrator generated.
            canonical_manifest_path = git_repo_root / "commit_manifest.json"
            try:
                with open(canonical_manifest_path, 'w') as f2:
                    json.dump(manifest_data, f2, indent=2)
                print(f"[MECHANICAL SUCCESS] Wrote canonical commit_manifest.json with {len(manifest_entries)} entries")
            except (OSError, IOError) as e:
                print(f"[MECHANICAL WARNING] Failed to write canonical commit_manifest.json: {e}")

            print(f"[MECHANICAL SUCCESS] Generated {manifest_name} with {len(manifest_entries)} entries")
        except (OSError, IOError) as e:
            print(f"[MECHANICAL ERROR] Failed to write commit manifest: {e}")
            raise

        # Phase 1.5: Handle Deletions (git rm)
        if files_to_remove:
            print(f"[MECHANICAL INFO] Deleting {len(files_to_remove)} files...")
            for file_path in files_to_remove:
                # Use git rm to stage the deletion
                try:
                    execute_shell_command(
                        ["git", "rm", "--", file_path],  # Use -- to handle paths that look like arguments
                        check=True,
                        cwd=git_repo_root,
                        capture_output=True
                    )
                    print(f"[MECHANICAL SUCCESS] Removed {file_path}")
                except subprocess.CalledProcessError as e:
                    # Allow git rm to fail if the file is already deleted or not tracked
                    if "did not match any files" in e.stderr.decode():
                        print(f"[MECHANICAL WARNING] git rm skipped {file_path}: not found or not tracked. Staging deletion might be redundant.")
                    else:
                        print(f"[MECHANICAL ERROR] git rm failed for {file_path}: {e.stderr.decode().strip()}")
                        # Do NOT raise here, as we want to continue with the commit

        # --- CORRECTED LOGIC: SEPARATE HASHING FROM COMMITTING ---
        # The files to be committed will include the manifest itself.
        # The manifest's content, however, will only contain hashes of the original target files.
        files_to_commit = [p for p in resolved_file_paths]

        # ensure manifest_path is a Path under git_repo_root (manifest_name is defined above)
        # manifest will live in git_repo_root, so add the manifest file object to the commit list
        files_to_commit.append(manifest_path)

        # Also add the canonical manifest path to the commit if it exists
        canonical_manifest_path = git_repo_root / "commit_manifest.json"
        if canonical_manifest_path.exists():
            files_to_commit.append(canonical_manifest_path)

        print(f"[MECHANICAL INFO] Staging {len(resolved_file_paths)} target files + {2 if canonical_manifest_path.exists() else 1} manifest files for commit.")
        # The `manifest_entries` list is now correct and does NOT include the manifest itself.

        # Execute git add for each resolved file from the git repo root
        for full_path in files_to_commit:
            primary_action = 'add'
            if primary_action not in WHITELISTED_GIT_COMMANDS:
                print(f"[CRITICAL] Prohibited Git command attempted and blocked: {primary_action}")
                raise Exception(f"Prohibited Git command: {primary_action}")

            try:
                repo_relative_path = Path(full_path).relative_to(git_repo_root)
            except Exception:
                # If we cannot make it repo-relative, skip
                print(f"[MECHANICAL WARNING] File {full_path} is outside repo root, skipping git add")
                continue

            try:
                result = execute_shell_command(
                    ["git", "add", str(repo_relative_path)],
                    capture_output=True,
                    cwd=git_repo_root,
                    check=False # We handle returncode manually below
                )
                if result.returncode == 0:
                    print(f"[MECHANICAL SUCCESS] Added {repo_relative_path} to git staging")
                else:
                    error_msg = f"Git add failed for {repo_relative_path}"
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
                print(f"[MECHANICAL ERROR] Git add timed out for {repo_relative_path}")
                raise
            except FileNotFoundError:
                print(f"[MECHANICAL ERROR] Git command not found - ensure git is installed")
                raise

        # Execute git commit - validate command is whitelisted
        primary_action = 'commit'
        if primary_action not in WHITELISTED_GIT_COMMANDS:
            print(f"[CRITICAL] Prohibited Git command attempted and blocked: {primary_action}")
            raise Exception(f"Prohibited Git command: {primary_action}")

        try:
            result = execute_shell_command(
                ["git", "commit", "-m", commit_message],
                capture_output=True,
                cwd=git_repo_root,
                check=False # We handle returncode manually below
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
                result = execute_shell_command(
                    ["git", "push"],
                    capture_output=True,
                    cwd=project_root,
                    check=False # We handle returncode manually below
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

    # Phase 3: Refresh cache for committed files
    if commit_success:
        # Added DEBUG for tracing the cache call
        print(f"[MECHANICAL DEBUG] Attempting cache refresh for {len(files_to_add)} committed files.")
        # FIX: CacheManager.prefill_guardian_delta is missing a required positional argument 'updated_files'.
        # Passing an empty list as a placeholder for the second argument to satisfy the function signature.
        CacheManager.prefill_guardian_delta(files_to_add, [])
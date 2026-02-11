#!/usr/bin/env python3
"""
Workflow Manager (Orchestrator)
=====================================

Purpose:
    Core logic for the "Python Orchestrator" architecture (ADR-0030 v2/v3).
    Handles the "Enforcement Layer" checks that were previously done in Bash:
    - Git State (Dirty tree, Detached HEAD)
    - Context Alignment (Pilot vs New Branch)
    - Branch Creation & Naming (via next_number.py)
    - Context Manifest Initialization

    Acts as the single source of truth for "Start Workflow" logic.

Input:
    - Workflow Name
    - Target ID
    - Artifact Type

Output:
    - Exit Code 0: Success (Proceed)
    - Exit Code 1: Failure (Stop, with printed reason)

Key Functions:
    - start_workflow: Main entry point.
    - get_git_status: Validates repo state.
    - get_current_branch: Returns active branch.
    - generate_next_id: Wraps next_number.py.

Usage:
    from tools.orchestrator.workflow_manager import WorkflowManager
    mgr = WorkflowManager()
    mgr.start_workflow("codify", "MyTarget")

Related:
    - tools/cli.py (Consumer)
    - docs/ADRs/0030-workflow-shim-architecture.md
"""
import sys
import subprocess
import re
from pathlib import Path

# Tool Discovery & Retrieval Policy: Late Binding
# We need to resolve paths dynamically using the resolver utility if available.
try:
    from tools.investigate.utils.path_resolver import resolve_path
except ImportError:
    # Fallback for direct execution/testing
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from tools.investigate.utils.path_resolver import resolve_path

class WorkflowManager:
    """
    Orchestrates the setup and pre-flight checks for Agent Workflows.
    Replaces legacy Bash-based checks with robust Python logic.
    """
    
    def __init__(self):
        # Policy: Avoid hardcoded assumptions. Use Path Resolver.
        try:
            self.project_root = Path(resolve_path(".")) # Root
            self.investigate_utils = Path(resolve_path("tools/investigate/utils"))
            self.next_num_script = self.investigate_utils / "next_number.py"
            self.specs_dir = self.project_root / "specs"
            self.cli_script = Path(resolve_path("tools/cli.py"))
        except Exception as e:
            # Fallback for unit testing or pure python env without resolver context
            self.project_root = Path(__file__).resolve().parent.parent.parent
            self.next_num_script = self.project_root / "tools" / "investigate" / "utils" / "next_number.py"
            self.specs_dir = self.project_root / "specs"
            self.cli_script = self.project_root / "tools" / "cli.py"

    def run_command(self, cmd: list, cwd: Path = None, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Helper to run subprocess commands with error handling.
        
        Args:
            cmd: List of command arguments.
            cwd: Working directory (defaults to Project Root).
            capture_output: Whether to capture stdout/stderr.
            
        Returns:
            CompletedProcess object.
            
        Raises:
            RuntimeError: If command returns non-zero exit code.
        """
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd or self.project_root,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            # Re-raise with stderr for better context
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\nStderr: {e.stderr}")

    def get_git_status(self) -> dict:
        """
        Checks for dirty working tree and detached HEAD.
        
        Returns:
            Dict with 'status' ('clean', 'dirty', 'detached_head') and optional 'details'.
        """
        # 1. Check for Dirty Tree
        res = self.run_command(["git", "status", "--porcelain"])
        if res.stdout.strip():
            return {"status": "dirty", "details": res.stdout.strip()}
        
        # 2. Check for Detached HEAD
        # If symbolic-ref fails, it's widely likely detached
        try:
            self.run_command(["git", "symbolic-ref", "HEAD"], capture_output=True)
            detached = False
        except RuntimeError:
            detached = True
            
        if detached:
            return {"status": "detached_head", "details": "Not on any branch"}

        return {"status": "clean"}

    def get_current_branch(self) -> str:
        """Returns the current active branch name."""
        try:
            res = self.run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            return res.stdout.strip()
        except RuntimeError:
            return "unknown"

    def generate_next_id(self) -> str:
        """Calls next_number.py to get the next spec ID."""
        if not self.next_num_script.exists():
            raise FileNotFoundError(f"next_number.py not found at {self.next_num_script}")
            
        res = self.run_command([sys.executable, str(self.next_num_script), "--type", "spec"])
        return res.stdout.strip()

    def start_workflow(self, workflow_name: str, target_id: str, artifact_type: str = "generic") -> bool:
        """
        Main entry point (ADR-0030 Compliant).
        1. Checks Git State.
        2. Determines Context (Pilot vs New).
        3. Creates/Switches Branch (Strict Enforcement).
        4. Initializes Spec Directory & Templates.
        5. Initializes Manifest.
        
        Args:
            workflow_name: Name of the workflow (e.g. 'codify-form')
            target_id: Target Artifact ID.
            artifact_type: Type of artifact (default: 'generic')
            
        Returns:
            True if successful, False otherwise.
        """
        print(f"🚀 Initializing Workflow: {workflow_name} for Target: {target_id}")
        
        # 1. Pre-Flight: Git State
        state = self.get_git_status()
        if state["status"] != "clean":
            if state["status"] == "dirty":
                print(f"❌ Error: Git working tree is dirty. Please commit or stash changes.")
                print(f"Details:\n{state['details']}")
            elif state["status"] == "detached_head":
                print(f"❌ Error: Git is in Detached HEAD state. Please checkout a branch.")
            return False

        current_branch = self.get_current_branch()
        print(f"✅ Git State: Clean. Branch: {current_branch}")

        # 2. Spec Folder Detection & Identity
        # Search specs/ for an existing folder matching target_id
        existing_spec_folder = None
        if self.specs_dir.exists():
            for spec_folder in self.specs_dir.iterdir():
                if spec_folder.is_dir() and target_id.lower() in spec_folder.name.lower():
                    existing_spec_folder = spec_folder
                    break
        
        spec_id = None
        if existing_spec_folder:
            spec_id = existing_spec_folder.name.split('-')[0]
            print(f"✅ Existing Spec Detected: {existing_spec_folder.name} (ID: {spec_id})")
        else:
            # Need new ID
            try:
                spec_id = self.generate_next_id()
                print(f"🆕 New Spec ID Generated: {spec_id}")
            except Exception as e:
                print(f"❌ Failed to generate spec ID: {e}")
                return False

        # 3. Branch Enforcement Logic (The "One Spec = One Branch" Rule)
        expected_branch_prefix = f"spec/{spec_id}"
        
        # CASE A: Already on the correct branch
        if expected_branch_prefix in current_branch:
             print(f"✅ Context Aligned: You are on the correct branch '{current_branch}'.")
        
        # CASE B: On Main or Develop -> Force Switch/Create
        elif current_branch in ["main", "develop", "master"]:
            print(f"⚠️  You are on '{current_branch}'. Strict Policy requires a feature branch.")
            
            # Check for existing branch matching this spec ID
            res = self.run_command(["git", "branch", "--list", f"*{spec_id}*"])
            matching_branches = [b.strip().strip("* ") for b in res.stdout.splitlines() if b.strip()]
            
            if matching_branches:
                target_branch = matching_branches[0]
                print(f"🔄 Switching to existing spec branch: {target_branch}")
                self.run_command(["git", "checkout", target_branch])
            else:
                # Create NEW Branch
                new_branch = f"spec/{spec_id}-{target_id.lower()}"
                print(f"✨ Creating new feature branch: {new_branch}")
                self.run_command(["git", "checkout", "-b", new_branch])
        
        # CASE C: On a DIFFERENT Feature Branch -> ERROR
        else:
            # Check if this is a "fix/" or child branch for the same spec?
            if spec_id in current_branch:
                # Same spec ID, different suffix (e.g. fix/003-bug) -> Allow
                print(f"✅ Context Aligned: Child branch '{current_branch}' matches Spec ID {spec_id}.")
            else:
                # Different spec context entirely
                print(f"❌ Error: You are on '{current_branch}' but target is Spec {spec_id}.")
                print(f"    Please switch to 'main' or the correct spec branch first.")
                return False

        # 4. Spec Directory Bootstrap
        # Now that we are on the branch, ensure the folder exists
        if not existing_spec_folder:
            new_spec_folder_name = f"{spec_id}-{target_id.lower()}"
            new_spec_path = self.specs_dir / new_spec_folder_name
            try:
                new_spec_path.mkdir(parents=True, exist_ok=True)
                print(f"📂 Created Spec Directory: {new_spec_path}")
                existing_spec_folder = new_spec_path # Update ref
                
                # COPY TEMPLATES Logic
                # Copy standard spec.md, plan.md, tasks.md?
                # Or copy SOP specific ones?
                # For now, we will rely on `speckit` commands to populate, 
                # BUT we can drop a README or the SOP if we identify it.
                # COPY TEMPLATES Logic
                # Copy sanctuary-start.md to the new folder
                template_path = self.project_root / ".agent" / "templates" / "workflow" / "workflow-start-template.md"
                dest_path = new_spec_path / "sanctuary-start.md"
                
                if template_path.exists():
                    dest_path.write_text(template_path.read_text())
                    print(f"📄 Created {dest_path.name} from template")
                else:
                    print(f"⚠️ Template not found: {template_path}")
                
                # Copy Scratchpad Template
                scratchpad_tpl = self.project_root / ".agent" / "templates" / "workflow" / "scratchpad-template.md"
                scratchpad_dest = new_spec_path / "scratchpad.md"
                if scratchpad_tpl.exists():
                    scratchpad_dest.write_text(scratchpad_tpl.read_text())
                    print(f"📄 Created {scratchpad_dest.name} from template")

                # Copy Retrospective Template (Deterministic Lifecycle)
                retro_tpl = self.project_root / ".agent" / "templates" / "workflow" / "workflow-retrospective-template.md"
                retro_dest = new_spec_path / "retrospective.md"
                if retro_tpl.exists():
                    retro_dest.write_text(retro_tpl.read_text())
                    print(f"📄 Created {retro_dest.name} from template")

                # Copy Workflow End Template (Deterministic Lifecycle)
                end_tpl = self.project_root / ".agent" / "templates" / "workflow" / "workflow-end-template.md"
                end_dest = new_spec_path / "sanctuary-end.md"
                if end_tpl.exists():
                    end_dest.write_text(end_tpl.read_text())
                    print(f"📄 Created {end_dest.name} from template")

                # Copy Core Spec/Plan/Tasks Templates (if not handled by Manifest)
                # This ensures the folder is fully populated 
                for doc, tpl_name in [("spec.md", "spec-template.md"), ("plan.md", "plan-template.md"), ("tasks.md", "tasks-template.md")]:
                     tpl = self.project_root / ".agent" / "templates" / "workflow" / tpl_name
                     dest = new_spec_path / doc
                     if tpl.exists() and not dest.exists():
                         dest.write_text(tpl.read_text())
                         print(f"📄 Created {dest.name} from template") 
            except Exception as e:
                print(f"❌ Failed to create spec directory: {e}")
                return False

        # 5. Context Manifest Initialization
        print(f"📦 Initializing Context Bundle...")
        try:
            self.run_command([
                sys.executable, str(self.cli_script), 
                "manifest", "init", 
                "--bundle-title", target_id,
                "--type", artifact_type
            ])
            print("✅ Context Manifest Initialized.")
        except Exception as e:
             print(f"⚠️ Warning: Context init failed (Manifest might already exist). Continuing.")

        print(f"\n🎉 Workflow '{workflow_name}' Ready. Execution passed to Agent.")
        return True

    def run_retrospective(self) -> bool:
        """
        Calculates the active spec and initializes the retrospective artifact.
        Runs proof check, then INTERACTIVELY prompts user for feedback.
        """
        print(f"🕵️  Initializing Retrospective...")
        
        # 1. Identify Active Spec based on branch or CWD
        current_branch = self.get_current_branch()
        spec_id = None
        
        if "spec/" in current_branch:
            # Extract spec ID from branch "spec/0001-foo"
            try:
                spec_id = current_branch.split("spec/")[1].split("-")[0]
            except IndexError:
                pass
        
        target_spec_folder = None
        if spec_id:
            # Find folder
            for f in self.specs_dir.iterdir():
                if f.is_dir() and f.name.startswith(spec_id):
                    target_spec_folder = f
                    break
        
        if not target_spec_folder:
            print("❌ Error: Could not determine active Spec context from branch.")
            return False
            
        print(f"✅ Active Spec Context: {target_spec_folder.name}")
        
        # 2. RUN PROOF CHECK
        print(f"\n🔍 Running Proof Check...")
        proof_result = self.run_proof_check(target_spec_folder)
        
        if proof_result["unchanged"]:
            # (Blocking logic remains same as existing code...)
            print("\n" + "="*60)
            print("🛑 PROOF CHECK FAILED - VERIFICATION REQUIRED")
            print("="*60)
            print("Files checked as modified but found UNCHANGED:")
            for item in proof_result["unchanged"]:
                print(f"   ❌ {item['path']}")
            print("\nDO NOT PROCEED until this check passes.")
            print("="*60)
            return False
        
        print(f"✅ Proof Check Passed: {len(proof_result['modified'])} file(s) verified.")
        
        # 3. INTERACTIVE FEEDBACK (Part A)
        print("\n" + "="*50)
        print("🧠 Protocol 128: Interactive Retrospective")
        print("="*50)
        print("Please answer the following questions to close the learning loop:\n")
        
        a1 = input("1. What went well? (Observation): ").strip() or "N/A"
        a2 = input("2. What was frustrating/confusing? (Observation): ").strip() or "N/A"
        a3 = input("3. Did Agent ignore any feedback? (If yes, details): ").strip() or "No"
        a4 = input("4. Suggestions for improvement?: ").strip() or "N/A"
        
        # 4. Load & Populate Template
        template_path = self.project_root / ".agent" / "templates" / "workflow" / "workflow-retrospective-template.md"
        dest_path = target_spec_folder / "retrospective.md"
        
        if dest_path.exists():
            print(f"⚠️  Retrospective file already exists at: {dest_path}")
            print(f"   (Skipping overwrite to preserve existing data)")
            return True
            
        if not template_path.exists():
             print(f"❌ Error: Template not found at {template_path}")
             return False
             
        try:
            content = template_path.read_text()
            
            # Inject Variables
            import datetime
            content = content.replace("[DATE]", str(datetime.date.today()))
            content = content.replace("[WORKFLOW_NAME]", target_spec_folder.name)
            
            # Inject Answers (Simple replacement of placeholders or appending)
            # The template has checkboxes like - [ ] [User observation]
            # We'll replace the first occurrence of appropriate placeholders
            
            content = content.replace("[User observation]", a1, 1) # A1
            content = content.replace("[User observation]", a2, 1) # A2 - reused placeholder?
            # actually template uses same placeholder. Let's be robust.
            # Strategy: Replace specific text blocks if they exist, or just append answers?
            # Template:
            # ### A1. What went well for you?
            # - [ ] [User observation]
            
            # Re-read template structure in memory logic:
            # We will use regex to robustly replace sections if simple replace is risky.
            # But simple replace logic:
            # 1st [User observation] -> A1
            # 2nd [User observation] -> A2
            # [Details] -> A3
            # [User suggestion] -> A4
            
            # Let's start fresh with the replacements
            content = template_path.read_text()
            content = content.replace("[DATE]", str(datetime.date.today()))
            content = content.replace("[WORKFLOW_NAME]", target_spec_folder.name)
            
            # Replace sequentially
            content = content.replace("[User observation]", a1, 1)
            content = content.replace("[User observation]", a2, 1)
            content = content.replace("[Details]", a3, 1)
            content = content.replace("[User suggestion]", a4, 1)
            
            # Auto-fill Files List
            file_list_str = ""
            for m in proof_result["modified"]:
                file_list_str += f"- [x] `{m['path']}`\n"
            
            content = content.replace("- [ ] `path/to/file.py`", file_list_str)

            dest_path.write_text(content)
            print(f"\n✅ Created Retrospective Artifact: {dest_path}")
            print("   Part A (User) is populated.")
            print("   👉 Part B (Agent Self-Assessment): Agent will fill this now (simulated).")
            return True
        except Exception as e:
            print(f"❌ Error writing file: {e}")
            return False

    def run_proof_check(self, spec_dir: Path) -> dict:
        """
        Scan spec.md, plan.md, tasks.md for file references
        and verify each has been modified vs origin/main.
        """
        import re
        
        results = {
            "modified": [],
            "unchanged": [],
            "not_found": [],
            "errors": []
        }
        
        all_refs = set()
        
        # Scan spec artifacts for file references
        for artifact in ["spec.md", "plan.md", "tasks.md"]:
            artifact_path = spec_dir / artifact
            if artifact_path.exists():
                content = artifact_path.read_text()
                
                # Extract backticked paths (e.g., `tools/cli.py`)
                backtick_pattern = r'`([^`]+\.(md|py|sh|json|js|ts|yaml|yml))`'
                for match in re.findall(backtick_pattern, content):
                    all_refs.add(match[0])
        
        print(f"   Found {len(all_refs)} file references")
        
        # Check each file for modifications
        for ref in sorted(all_refs):
            # Normalize path
            if ref.startswith("/"):
                ref = ref[1:]
            
            full_path = self.project_root / ref
            
            if not full_path.exists():
                results["not_found"].append({"path": ref})
                continue
            
            # Check git diff
            try:
                result = self.run_command(
                    ["git", "diff", "--stat", "origin/main", "--", str(full_path)],
                    capture_output=True
                )
                
                if result and result.strip():
                    results["modified"].append({"path": ref, "details": result.strip()})
                else:
                    results["unchanged"].append({"path": ref})
                    
            except Exception as e:
                results["errors"].append({"path": ref, "details": str(e)})
        
        return results

    def end_workflow(self, message: str, files: list) -> bool:
        """
        Completes the workflow by committing and pushing changes.
        """
        print(f"🏁 Finalizing Workflow...")
        
        # 1. Check Git Status
        state = self.get_git_status()
        current_branch = self.get_current_branch()
        
        # 2. Add Files
        if not files:
            print("⚠️  No files specified. Adding ALL files (tracked + untracked).")
            self.run_command(["git", "add", "."])
        else:
            print(f"📦 Staging {len(files)} files...")
            for f in files:
                self.run_command(["git", "add", f])

        # 1.5 Enforce Workflow End Checklist
        spec_id = None
        if "spec/" in current_branch:
             try: spec_id = current_branch.split("spec/")[1].split("-")[0]
             except: pass
        
        target_spec_folder = None
        if spec_id:
             for f in self.specs_dir.iterdir():
                  if f.is_dir() and f.name.startswith(spec_id):
                       target_spec_folder = f
                       break
        
        if target_spec_folder:
             # 1.5a Enforce Retrospective (Protocol 128)
             retro_path = target_spec_folder / "retrospective.md"
             if not retro_path.exists():
                 print(f"\n🛑 BLOCKING: No Retrospective Artifact found.")
                 print(f"   path: {retro_path}")
                 print("   The Universal Hybrid Workflow requires a retrospective before closure.")
                 print("\n👉 ACTION REQUIRED: Run '/sanctuary-retrospective' first.\n")
                 return False

             # 1.5b Enforce Workflow End Checklist
             checklist_path = target_spec_folder / "sanctuary-end.md"
             if not checklist_path.exists():
                   # Copy Template
                   tpl_path = self.project_root / ".agent" / "templates" / "workflow" / "workflow-end-template.md"
                   if tpl_path.exists():
                       checklist_path.write_text(tpl_path.read_text())
                       print(f"🛑 Checklist Created: {checklist_path}")
                       print("   Please review/complete this checklist, then run 'workflow end' again.")
                       # Stage it so it's included next time
                       self.run_command(["git", "add", str(checklist_path)])
                       return False # Stop to let user review

                
        # 3. Commit
        print(f"💾 Committing with message: '{message}'")
        try:
            self.run_command(["git", "commit", "-m", message])
        except Exception:
             print("⚠️  Commit failed (nothing to commit?). Proceeding.")
             
        # 4. Push
        print(f"🚀 Pushing branch '{current_branch}'...")
        try:
            self.run_command(["git", "push", "origin", current_branch])
            print(f"✅ Workflow Completed. Branch '{current_branch}' pushed.")
            
            print("\n" + "="*50)
            print("👉 NEXT STEPS:")
            print(f"   1. Create Pull Request for branch '{current_branch}'")
            print("   2. Wait for CI/CD & Review")
            print("   3. Merge to main")
            print("="*50 + "\n")

            return True
        except Exception as e:
            print(f"❌ Push failed: {e}")
            return False

    def end_workflow_with_confirmation(self, message: str, files: list, force: bool = False) -> bool:
        """
        Completes the workflow with explicit user confirmation (unless --force).
        """
        if not force:
            print("\n" + "="*50)
            print("⚠️  CONFIRMATION REQUIRED")
            print("="*50)
            print(f"Commit Message: {message}")
            print(f"Files: {files if files else 'All tracked'}")
            print(f"Branch: {self.get_current_branch()}")
            print("="*50)
            
            response = input("\n🔐 Type 'yes' to confirm push, or anything else to cancel: ").strip().lower()
            if response != 'yes':
                print("❌ Push cancelled by user.")
                return False
        
        success = self.end_workflow(message, files)
        if success:
             print("\n💡 TIP: Once merged, run: /workflow-cleanup")
        return success

    def cleanup_workflow(self, force: bool = False) -> bool:
        """
        Step 4: Post-Merge Cleanup.
        1. Checkout main
        2. Pull origin main
        3. Delete local feature branch
        """
        current_branch = self.get_current_branch()
        if current_branch in ["main", "master", "develop"]:
             print(f"❌ You are already on '{current_branch}'. Cleanup is for feature branches.")
             return False
        
        print(f"\n🧹 Cleanup Routine for branch '{current_branch}'")
        print("   1. Checkout 'main'")
        print("   2. Pull latest 'main'")
        print("   3. Delete local branch")
        
        if not force:
             print("\n⚠️  PREREQUISITE: Ensure your Pull Request is MERGED first.")
             resp = input("❓ Ready to proceed? (yes/no): ").strip().lower()
             if resp != 'yes':
                  print("❌ Aborting cleanup.")
                  return False

        print(f"\n🔄 Switching to 'main'...")
        try:
             self.run_command(["git", "checkout", "main"])
        except Exception as e:
             print(f"❌ Failed to checkout main: {e}")
             return False

        print(f"⬇️  Pulling 'origin main'...")
        try:
             self.run_command(["git", "pull", "origin", "main"])
        except Exception as e:
             print(f"❌ Failed to pull main: {e}")
             # We continue, as we might still want to delete the branch, but safest to warn?
             # Actually, if pull fails, we might be out of sync. But let's verify branch deletion.

        print(f"🗑️  Deleting branch '{current_branch}'...")
        try:
             # Try safe delete first (checks merge status)
             self.run_command(["git", "branch", "-d", current_branch])
             print(f"✅ Branch '{current_branch}' deleted successfully.")
        except Exception:
             print(f"⚠️  Safe delete failed (maybe commit history mismatch?).")
             resp = input(f"❓ Force delete '{current_branch}'? (yes/no): ").strip().lower()
             if resp == 'yes':
                  try:
                       self.run_command(["git", "branch", "-D", current_branch])
                       print(f"✅ Branch '{current_branch}' force deleted.")
                  except Exception as e:
                       print(f"❌ Force delete failed: {e}")
                       return False
             else:
                  print("❌ Skipped deletion.")
                  return False
        
        return True

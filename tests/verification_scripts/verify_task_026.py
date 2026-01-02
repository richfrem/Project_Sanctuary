
import sys
import json
import os
import shutil
from pathlib import Path
from mcp_servers.orchestrator.tools.cognitive import create_cognitive_task
from mcp_servers.orchestrator.tools.mechanical import create_git_commit_task

def verify_task_026():
    print("--- Starting Task #026 Verification ---")
    
    # Setup
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    orchestrator_dir = project_root / "council_orchestrator"
    if orchestrator_dir.exists():
        shutil.rmtree(orchestrator_dir)
    
    # 1. Test Cognitive Task Creation
    print("\n1. Testing Cognitive Task Creation...")
    result = create_cognitive_task(
        description="Test cognitive task",
        output_path="WORK_IN_PROGRESS/test_output.md",
        max_rounds=3
    )
    
    if result["status"] == "success":
        print("   [SUCCESS] Task created.")
        cmd_file = Path(result["command_file"])
        if cmd_file.exists():
            print(f"   [SUCCESS] command.json exists at {cmd_file}")
            with open(cmd_file, "r") as f:
                data = json.load(f)
                if data["task_description"] == "Test cognitive task":
                    print("   [SUCCESS] Content verified.")
                else:
                    print("   [FAIL] Content mismatch.")
        else:
             print("   [FAIL] command.json not found.")
    else:
        print(f"   [FAIL] Task creation failed: {result.get('error')}")

    # 2. Test Safety Guardrails (Protected File)
    print("\n2. Testing Safety Guardrails (Protected File)...")
    result = create_git_commit_task(
        files=["01_PROTOCOLS/95_The_Commandable_Council_Protocol.md"],
        message="feat: modify protocol",
        description="Attempt to modify protocol"
    )
    
    if result["status"] == "error" and "protected path" in result["error"].lower():
        print(f"   [SUCCESS] Blocked protected file modification: {result['error']}")
    else:
        print(f"   [FAIL] Should have blocked protected file. Result: {result}")

    # 3. Test Safety Guardrails (Invalid Commit Message)
    print("\n3. Testing Safety Guardrails (Invalid Commit Message)...")
    result = create_git_commit_task(
        files=["tasks/backlog/test.md"],
        message="bad message",
        description="Attempt bad commit"
    )
    
    if result["status"] == "error" and "conventional commit" in result["error"].lower():
        print(f"   [SUCCESS] Blocked invalid commit message: {result['error']}")
    else:
        print(f"   [FAIL] Should have blocked invalid message. Result: {result}")

    # Cleanup
    # if orchestrator_dir.exists():
    #     shutil.rmtree(orchestrator_dir)

if __name__ == "__main__":
    verify_task_026()

"""
End-to-end workflow test for Task MCP
Tests the complete workflow: create → update → move → search
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.task.operations import TaskOperations
from mcp_servers.task.models import TaskStatus, TaskPriority


def test_complete_workflow():
    """Test complete task workflow"""
    
    print("🧪 Starting End-to-End Workflow Test\n")
    
    # Initialize operations
    task_ops = TaskOperations(project_root)
    
    # Step 1: Create a test task
    print("Step 1: Creating test task...")
    result = task_ops.create_task(
        title="E2E Test Task - MCP Server Validation",
        objective="Validate the Task MCP server end-to-end workflow",
        deliverables=[
            "Create task successfully",
            "Update task metadata",
            "Move task through statuses",
            "Search and retrieve task"
        ],
        acceptance_criteria=[
            "Task created in backlog",
            "Task updated with new priority",
            "Task moved to in-progress",
            "Task searchable and retrievable"
        ],
        priority=TaskPriority.HIGH,
        status=TaskStatus.BACKLOG,
        lead="Antigravity Test Suite",
        notes="This is an automated end-to-end test"
    )
    
    assert result.status == "success", f"Create failed: {result.message}"
    task_number = result.task_number
    print(f"✅ Task #{task_number:03d} created successfully")
    print(f"   File: {result.file_path}\n")
    
    # Step 2: Retrieve the task
    print("Step 2: Retrieving task...")
    task = task_ops.get_task(task_number)
    assert task is not None, "Task not found"
    assert task["title"] == "E2E Test Task - MCP Server Validation"
    print(f"✅ Task retrieved: {task['title']}")
    print(f"   Status: {task['status']}\n")
    
    # Step 3: Update task priority
    print("Step 3: Updating task priority to CRITICAL...")
    result = task_ops.update_task(
        task_number=task_number,
        updates={"priority": TaskPriority.CRITICAL}
    )
    assert result.status == "success", f"Update failed: {result.message}"
    print(f"✅ Task updated successfully\n")
    
    # Step 4: Move task to in-progress
    print("Step 4: Moving task to IN-PROGRESS...")
    result = task_ops.update_task_status(
        task_number=task_number,
        new_status=TaskStatus.IN_PROGRESS,
        notes="Starting E2E test validation"
    )
    assert result.status == "success", f"Status update failed: {result.message}"
    assert "in-progress" in result.file_path
    print(f"✅ Task moved to in-progress")
    print(f"   New location: {result.file_path}\n")
    
    # Step 5: Search for the task
    print("Step 5: Searching for task...")
    results = task_ops.search_tasks("E2E Test Task")
    assert len(results) > 0, "Task not found in search"
    assert results[0]["number"] == task_number
    print(f"✅ Task found in search")
    print(f"   Matches: {len(results[0]['matches'])} lines\n")
    
    # Step 6: List tasks in progress
    print("Step 6: Listing in-progress tasks...")
    tasks = task_ops.list_tasks(status=TaskStatus.IN_PROGRESS)
    task_numbers = [t["number"] for t in tasks]
    assert task_number in task_numbers, "Task not in in-progress list"
    print(f"✅ Task found in in-progress list")
    print(f"   Total in-progress tasks: {len(tasks)}\n")
    
    # Step 7: Move to done
    print("Step 7: Moving task to DONE...")
    result = task_ops.update_task_status(
        task_number=task_number,
        new_status=TaskStatus.COMPLETE,
        notes="E2E test completed successfully"
    )
    assert result.status == "success"
    assert "done" in result.file_path
    print(f"✅ Task completed and moved to done")
    print(f"   Final location: {result.file_path}\n")
    
    # Final verification
    print("Final Verification:")
    final_task = task_ops.get_task(task_number)
    assert final_task["status"] == "complete"
    assert final_task["priority"] == "Critical"
    print(f"✅ All assertions passed!")
    print(f"   Task #{task_number:03d}: {final_task['title']}")
    print(f"   Status: {final_task['status']}")
    print(f"   Priority: {final_task['priority']}")
    
    print("\n🎉 End-to-End Workflow Test PASSED!")
    print(f"\nTask #{task_number:03d} can be found at:")
    print(f"   {project_root / result.file_path}")
    
    return task_number


if __name__ == "__main__":
    try:
        task_num = test_complete_workflow()
        print(f"\n✅ SUCCESS: Task #{task_num:03d} created and validated")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

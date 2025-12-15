"""
Task MCP E2E Tests - Protocol Verification
==========================================

Verifies all tools via JSON-RPC protocol against the real Task server.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/task/e2e/test_task_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool                  | Type  | Description              |
|-----------------------|-------|--------------------------|
| create_task           | WRITE | Create task file         |
| get_task              | READ  | Get task details         |
| list_tasks            | READ  | List tasks               |
| search_tasks          | READ  | Search tasks             |
| update_task           | WRITE | Update metadata          |
| update_task_status    | WRITE | Move file/status         |

"""
import pytest
import os
import re
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestTaskE2E(BaseE2ETest):
    SERVER_NAME = "task"
    SERVER_MODULE = "mcp_servers.task.server"

    def test_task_lifecycle(self, mcp_client):
        """Test full cycle: Create -> Get -> List -> Search -> Update -> Status -> Cleanup"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        assert "create_task" in names

        # 2. Create Task
        # Use high number 999 to avoid conflict.
        task_num = 999
        title = "E2E Test Task"
        
        create_res = mcp_client.call_tool("create_task", {
            "title": title,
            "objective": "Verify E2E",
            "deliverables": ["Report"],
            "acceptance_criteria": ["Passed"],
            "task_number": task_num,
            "status": "todo"
        })
        create_text = create_res.get("content", [])[0]["text"]
        print(f"\n🆕 create_task: {create_text}")
        
        # Parse path
        # Output: "Created Task 999: TASKS/todo/999_E2E_Test_Task.md"
        match = re.search(r"Created Task (\d+): (.+)", create_text)
        assert match, f"Failed to parse creation response: {create_text}"
        
        t_num = int(match.group(1))
        assert t_num == task_num
        rel_path = match.group(2).strip()
        
        # Handle path
        if Path(rel_path).is_absolute():
            task_path = Path(rel_path)
        else:
            task_path = PROJECT_ROOT / rel_path
            
        print(f"   Task #{t_num} at {task_path}")

        try:
            # 3. Get Task
            get_res = mcp_client.call_tool("get_task", {"task_number": task_num})
            get_text = get_res.get("content", [])[0]["text"]
            assert title in get_text
            assert "Status: todo" in get_text
            print("📄 get_task: Verified content")

            # 4. List Tasks
            list_res = mcp_client.call_tool("list_tasks", {"status": "todo"})
            list_text = list_res.get("content", [])[0]["text"]
            assert f"{task_num:03d}" in list_text or f"{task_num}" in list_text
            print(f"📋 list_tasks: Found task in todo")

            # 5. Search
            search_res = mcp_client.call_tool("search_tasks", {"query": "Verify E2E"})
            search_text = search_res.get("content", [])[0]["text"]
            assert f"{task_num:03d}" in search_text or f"{task_num}" in search_text
            print(f"🔍 search_tasks: Found task")

            # 6. Update Task Metadata
            update_res = mcp_client.call_tool("update_task", {
                "task_number": task_num,
                "updates": {"priority": "Critical"}
            })
            update_text = update_res.get("content", [])[0]["text"]
            print(f"🔄 update_task: {update_text}")
            assert "Updated Task" in update_text

            # 7. Update Status (Moves file)
            # todo -> complete
            status_res = mcp_client.call_tool("update_task_status", {
                "task_number": task_num,
                "new_status": "complete",
                "notes": "Completed by E2E"
            })
            status_text = status_res.get("content", [])[0]["text"]
            print(f"🚚 update_task_status: {status_text}")
            
            # Verify status update
            get_res_2 = mcp_client.call_tool("get_task", {"task_number": task_num})
            get_text_2 = get_res_2.get("content", [])[0]["text"]
            assert "Status: complete" in get_text_2
            
        finally:
            # 8. Cleanup
            # Determine filename from original path logic or glob
            # Original: task_path.
            # Check likely locations
            possible_paths = [
                task_path,
                PROJECT_ROOT / "TASKS/todo" / task_path.name,
                PROJECT_ROOT / "TASKS/done" / task_path.name,
                PROJECT_ROOT / "TASKS/backlog" / task_path.name,
                PROJECT_ROOT / "TASKS/archive" / task_path.name
            ]
            
            cleaned = False
            for p in possible_paths:
                if p.exists():
                    os.remove(p)
                    print(f"🧹 Cleaned up {p}")
                    cleaned = True
            
            if not cleaned:
                # Fallback glob cleanup for safety
                found = list(PROJECT_ROOT.glob(f"TASKS/*/*{task_num}*.md"))
                for f in found:
                    os.remove(f)
                    print(f"🧹 Cleaned up found file: {f}")

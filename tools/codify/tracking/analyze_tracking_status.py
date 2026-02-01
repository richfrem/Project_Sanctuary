"""
tools/codify/tracking/analyze_tracking_status.py
================================================

Purpose:
    Generates a summary report of task/spec completion progress.
    Shows completed vs pending tasks for project management dashboards.

Input:
    - .agent/learning/task_tracking.json (or tasks/ directory)

Usage:
    python tools/codify/tracking/analyze_tracking_status.py
"""
import json
import os
from pathlib import Path

# Project Sanctuary task tracking
TRACKING_FILE = Path('.agent/learning/task_tracking.json')
TASKS_DIR = Path('tasks')

def analyze_status():
    """Analyze task completion status from tracking file or tasks directory."""
    
    tasks = {}
    
    # Try loading from tracking file first
    if TRACKING_FILE.exists():
        try:
            with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tasks = data.get('tasks', {})
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    
    # Fallback: scan tasks directory structure
    if not tasks and TASKS_DIR.exists():
        for status_dir in ['in-progress', 'blocked', 'done']:
            status_path = TASKS_DIR / status_dir
            if status_path.exists():
                for task_file in status_path.glob('*.md'):
                    task_id = task_file.stem
                    tasks[task_id] = {
                        'name': task_id.replace('-', ' ').title(),
                        'status': status_dir
                    }
    
    if not tasks:
        print("No tasks found. Create tasks in tasks/ directory or .agent/learning/task_tracking.json")
        return

    total = len(tasks)
    completed_count = 0
    pending_list = []

    print(f"\n{'='*60}")
    print(f"Project Sanctuary Task Tracking Summary")
    print(f"{'='*60}")

    for task_id, info in tasks.items():
        status = info.get('status', 'pending').lower()
        if status in ['done', 'completed', 'closed']:
            completed_count += 1
        else:
            pending_list.append({
                'id': task_id, 
                'name': info.get('name', 'Unknown'),
                'status': status
            })

    print(f"Total Tasks Tracked: {total}")
    print(f"✅ Completed:         {completed_count}")
    print(f"⚠️  Pending/In-Progress: {len(pending_list)}")
    print(f"{'='*60}\n")
    
    if pending_list:
        print("Tasks Requiring Attention:")
        print(f"{'-'*80}")
        print(f"{'Task ID':<25} | {'Status':<15} | {'Description'}")
        print(f"{'-'*80}")
        for item in pending_list:
            print(f"{item['id']:<25} | {item['status']:<15} | {item['name']}")
        print(f"{'-'*80}\n")

if __name__ == "__main__":
    analyze_status()

"""
tools/codify/tracking/generate_todo_list.py
===========================================

Purpose:
    Creates a prioritized TODO list of specs/tasks pending completion.
    Bubbles up blocked and in-progress items based on workflow stage.

Output:
    - TODO_PENDING_TASKS.md

Usage:
    python tools/codify/tracking/generate_todo_list.py
"""
import json
from pathlib import Path
from datetime import datetime

TASKS_DIR = Path('tasks')
OUTPUT_FILE = Path('TODO_PENDING_TASKS.md')
SPECS_DIR = Path('forge')

def generate_todo():
    """Generate prioritized TODO list from tasks and specs."""
    
    pending = []
    
    # Scan tasks directory
    if TASKS_DIR.exists():
        for status_dir in ['in-progress', 'blocked']:
            status_path = TASKS_DIR / status_dir
            if status_path.exists():
                for task_file in status_path.glob('*.md'):
                    task_id = task_file.stem
                    
                    # Determine priority
                    priority_score = 2 if status_dir == 'blocked' else 1
                    priority_label = "🔥 BLOCKED" if status_dir == 'blocked' else "⚡ IN-PROGRESS"
                    
                    pending.append({
                        'id': task_id,
                        'name': task_id.replace('-', ' ').title(),
                        'category': 'task',
                        'priority_score': priority_score,
                        'priority_label': priority_label
                    })
    
    # Scan forge for incomplete specs
    if SPECS_DIR.exists():
        for spec_dir in SPECS_DIR.iterdir():
            if spec_dir.is_dir():
                spec_file = spec_dir / 'spec.md'
                tasks_file = spec_dir / 'tasks.md'
                
                # Check if spec exists but tasks don't
                if spec_file.exists() and not tasks_file.exists():
                    pending.append({
                        'id': spec_dir.name,
                        'name': f"Spec: {spec_dir.name}",
                        'category': 'spec',
                        'priority_score': 1,
                        'priority_label': "📋 NEEDS TASKS"
                    })

    if not pending:
        print("✅ No pending tasks found. All clear!")
        return

    # Sort by Priority (Desc), then category, then ID
    pending.sort(key=lambda x: (-x['priority_score'], x['category'], x['id']))

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Pending Tasks To-Do List (Prioritized)\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Total Pending:** {len(pending)}\n\n")
        f.write("Blocked items and incomplete specs are bubbled to the top.\n\n")
    
        current_priority = -1
        
        for item in pending:
            if item['priority_score'] != current_priority:
                current_priority = item['priority_score']
                f.write(f"\n### {item['priority_label']}\n")
                f.write(f"| ID | Category | Description |\n")
                f.write(f"| :--- | :--- | :--- |\n")
            
            f.write(f"| **{item['id']}** | {item['category']} | {item['name']} |\n")

    print(f"✅ Generated {OUTPUT_FILE} with {len(pending)} items (Prioritized).")

if __name__ == "__main__":
    generate_todo()

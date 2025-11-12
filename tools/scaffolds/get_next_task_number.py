# tools/scaffolds/get_next_task_number.py
import os
from pathlib import Path
import re

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_DIRS = [
    PROJECT_ROOT / "TASKS",
    PROJECT_ROOT / "TASKS" / "backlog",
    PROJECT_ROOT / "TASKS" / "todo",
    PROJECT_ROOT / "TASKS" / "in-progress",
]

def get_next_task_number():
    """
    Scans all task directories to find the highest existing task number
    and returns the next sequential number as a zero-padded three-digit string.
    """
    highest_num = 0
    task_file_pattern = re.compile(r"^(\d{3})_.*\.md$")

    for directory in TASKS_DIRS:
        if not directory.exists():
            continue
        
        for filename in os.listdir(directory):
            match = task_file_pattern.match(filename)
            if match:
                num = int(match.group(1))
                if num > highest_num:
                    highest_num = num

    next_num = highest_num + 1
    return f"{next_num:03d}"

def main():
    """Main function to print the next available task number."""
    next_task_number = get_next_task_number()
    print(next_task_number)

if __name__ == "__main__":
    main()

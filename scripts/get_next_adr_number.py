# tools/scaffolds/get_next_adr_number.py
import os
from pathlib import Path
import re

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ADRS_DIR = PROJECT_ROOT / "ADRs"

def get_next_adr_number():
    """
    Scans the ADRs directory to find the highest existing ADR number
    and returns the next sequential number as a zero-padded three-digit string.
    """
    highest_num = 0
    adr_file_pattern = re.compile(r"^(\d{3})_.*\.md$")

    if not ADRS_DIR.exists():
        # If ADRs directory doesn't exist yet, start from 001
        return "001"

    for filename in os.listdir(ADRS_DIR):
        match = adr_file_pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > highest_num:
                highest_num = num

    next_num = highest_num + 1
    return f"{next_num:03d}"

def main():
    """Main function to print the next available ADR number."""
    next_adr_number = get_next_adr_number()
    print(next_adr_number)

if __name__ == "__main__":
    main()
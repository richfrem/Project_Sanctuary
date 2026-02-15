#!/usr/bin/env python3
"""
fix_analysis_links.py (CLI)
=====================================

Purpose:
    Fixes legacy analysis path references.

Layer: Curate / Utilities

Usage Examples:
    python tools/curate/link-checker/fix_analysis_links.py --help

Supported Object Types:
    - Generic

CLI Arguments:
    (None detected)

Input Files:
    - (See code)

Output:
    - (See code)

Key Functions:
    - fix_analysis_files(): No description.

Script Dependencies:
    (None detected)

Consumed by:
    (Unknown)
"""
import os

def fix_analysis_files():
    # Calculate path relative to this script (scripts/link-checker/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.join(project_root, "legacy-system", "previous-analysis-of-forms-and-business-rules", "business-rules", "by-form")
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    files = [f for f in os.listdir(base_dir) if f.endswith('.md')]
    
    replacements = [
        # Fix over-nested paths (idempotency fix)
        ("../../../../../tools", "../../../../tools"),
        ("../../../../../legacy-system", "../../../../legacy-system"),
        
        # Original logic (kept for other files possibly)
        ("../../legacy-system", "../../../legacy-system"),
        ("../../tools", "../../../tools"),
        ("../PROJECT_Access_Control_Overview.md", "../overview/PROJECT_Access_Control_Overview.md"),
        ("../Access-Control-Overview.md", "../overview/PROJECT_Access_Control_Overview.md"),
        ("../PROJECT_Cross_Application_Access_Control_Overview.md", "../overview/PROJECT_Cross_Application_Access_Control_Overview.md"),
        ("../FineGrainedAccessControl", "../fine-grained-access"),
        ("MenuConfig Menu Item Rules.csv", "menu-item-rules.csv"),
        ("MenuConfig%20Menu%20Item%20Rules.csv", "menu-item-rules.csv"),
        # Fix missing FormSpecificOutputs
        ("legacy-system/menuconfig-configuration/FormSpecificOutputs/", "legacy-system/menuconfig-configuration/"), 
        # Fix Prompts casing
        ("../Prompts/", "../../../tools/business-rule-extraction/prompts/"),
        # Remove Archive links? No, we can't easily regex-replace whole links here, just paths.
        ("modernization/archive/", "modernization/"),
    ]

    # Additional logic to remove broken links entirely would require regex.
    # We will stick to path corrections where possible.


    for filename in files:
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        for old, new in replacements:
            new_content = new_content.replace(old, new)
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed: {filename}")
        else:
            print(f"No changes: {filename}")

if __name__ == "__main__":
    fix_analysis_files()

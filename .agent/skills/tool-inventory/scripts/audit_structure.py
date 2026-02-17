#!/usr/bin/env python3
"""
Audit Plugin Structure
======================

Audits all plugins in `plugins/` to ensure they follow the standard structure:
- plugins/<plugin>/skills/<skill>/SKILL.md
- plugins/<plugin>/skills/<skill>/scripts/ (Optional but preferred over top-level)
- No top-level scripts/ in plugin root (unless exempt)

Usage:
    python3 plugins/tool-inventory/skills/tool-inventory/scripts/audit_structure.py
"""

import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.parent
PLUGINS_DIR = PROJECT_ROOT / "plugins"

# Exemptions (Legacy or Bridge plugins that might have top-level scripts)
# Ideally, we want 0 exemptions eventually.
TOP_LEVEL_SCRIPTS_EXEMPT = {
    # 'plugin-manager', # Should probably fix this too
}

def audit_plugin(plugin_path: Path):
    errors = []
    warnings = []
    
    plugin_name = plugin_path.name
    
    # 1. Check for 'skills' directory
    skills_dir = plugin_path / "skills"
    if not skills_dir.exists():
        errors.append(f"Missing 'skills/' directory.")
        return errors, warnings

    # 2. Check individual skills
    has_skills = False
    for skill_path in skills_dir.iterdir():
        if skill_path.is_dir():
            has_skills = True
            # Check SKILL.md
            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                errors.append(f"Skill '{skill_path.name}' missing SKILL.md")
            
            # Check for top-level scripts in skill
            # (Scripts should be in scripts/, not loose in skill root)
            for f in skill_path.glob("*.py"):
                 warnings.append(f"Skill '{skill_path.name}' has loose python script '{f.name}'. Move to scripts/?")

    if not has_skills:
        errors.append(f"'skills/' directory is empty.")

    # 3. Check for deprecated top-level directories
    if (plugin_path / "scripts").exists() and plugin_name not in TOP_LEVEL_SCRIPTS_EXEMPT:
        # Check if it's empty
        if any((plugin_path / "scripts").iterdir()):
             errors.append(f"Has top-level 'scripts/' directory (NON-STANDARD). Move to skills/<skill>/scripts/.")
    
    if (plugin_path / "docs").exists():
         warnings.append(f"Has top-level 'docs/' directory. Consider moving to skills/<skill>/references/.")

    return errors, warnings

def main():
    print(f"🏗️  Auditing Plugin Structure in {PLUGINS_DIR}...\n")
    
    results = {}
    
    for plugin_path in sorted(PLUGINS_DIR.iterdir()):
        if not plugin_path.is_dir(): continue
        if plugin_path.name.startswith("."): continue
        if plugin_path.name == "__pycache__": continue
        
        errors, warnings = audit_plugin(plugin_path)
        if errors or warnings:
            results[plugin_path.name] = {"errors": errors, "warnings": warnings}

    # Report
    if not results:
        print("✨ All plugins follow standard structure!")
        sys.exit(0)
        
    print(f"⚠️  Found issues in {len(results)} plugins:\n")
    
    error_count = 0
    
    for plugin, inconsistencies in results.items():
        if inconsistencies["errors"]:
            print(f"❌ {plugin}:")
            for e in inconsistencies["errors"]:
                print(f"   - {e}")
                error_count += 1
        
        if inconsistencies["warnings"]:
            if not inconsistencies["errors"]:
                print(f"⚠️  {plugin}:")
            for w in inconsistencies["warnings"]:
                print(f"   - {w}")
        print("")

    if error_count > 0:
        print(f"❌ Total Errors: {error_count}")
        sys.exit(1)
    else:
        print("✅ No critical errors found (only warnings).")
        sys.exit(0)

if __name__ == "__main__":
    main()

"""Fix all remaining hardcoded paths in archived and experimental files."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARDCODED = '/Users/richardfremmerlid/Projects/Project_Sanctuary'

files_to_fix = [
    '05_ARCHIVED_BLUEPRINTS/gardener_pytorch_rl_v1/chrysalis_awakening.py',
    '05_ARCHIVED_BLUEPRINTS/gardener_pytorch_rl_v1/gardener.py',
    '05_ARCHIVED_BLUEPRINTS/gardener_pytorch_rl_v1/environment.py',
    '05_ARCHIVED_BLUEPRINTS/gardener_pytorch_rl_v1/chrysalis_awakening_v2.py',
    'EXPERIMENTS/gardener_protocol37_experiment/chrysalis_awakening.py',
    'EXPERIMENTS/gardener_protocol37_experiment/gardener.py',
    'EXPERIMENTS/gardener_protocol37_experiment/environment.py',
]

for rel_path in files_to_fix:
    file_path = PROJECT_ROOT / rel_path
    if not file_path.exists():
        print(f'⊘ Skip: {rel_path} (not found)')
        continue
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        print(f'⊘ Skip: {rel_path} (encoding error)')
        continue
        
    if HARDCODED not in content:
        print(f'✓ Clean: {rel_path}')
        continue
    
    count_before = content.count(HARDCODED)
    
    # Fix default parameter values
    content = content.replace(
        f'= "{HARDCODED}"',
        '= None  # Computed from Path(__file__)'
    )
    content = content.replace(
        f'Path("{HARDCODED}")',
        'Path(__file__).resolve().parent.parent.parent'
    )
    
    count_after = content.count(HARDCODED)
    
    file_path.write_text(content, encoding='utf-8')
    print(f'✓ Fixed: {rel_path} ({count_before - count_after} occurrences)')

print('\n✅ All files processed!')

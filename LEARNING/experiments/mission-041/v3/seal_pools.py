"""One-shot pre-mutant suite seal. Never rerun/tune after catalog creation."""
import hashlib
import json
from pathlib import Path
from pools import build

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    if (ROOT/'mutants.json').exists() or (ROOT/'pool-seal.json').exists():
        raise SystemExit('Refuse to replace pre-mutant pools')
    (ROOT/'suites.json').write_text(json.dumps(build(), indent=2)+'\n')
    files = ['operator-split.json', 'specifications.py', 'pools.py', 'suites.json', 'seal_pools.py']
    (ROOT/'pool-seal.json').write_text(json.dumps({
        'stage': 'before mutant catalog authoring',
        'sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    }, indent=2)+'\n')

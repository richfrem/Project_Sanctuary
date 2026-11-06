"""Freeze and verify exact input inventory; no git commands or scoring."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
EXTERNAL = [
 'plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/counterexample_eval.py',
 'plugins/sanctuary-core/.apm/skills/evaluating-agent-work/tests/test_counterexample_eval.py',
 'plugins/sanctuary-core/.apm/skills/evaluating-agent-work/references/property-falsification.md',
 'LEARNING/experiments/mission-041/v2/errata.md']
# Only these phase-2 files may be added without invalidating input inventory.
OUTPUTS = {'phase2-results.json', 'phase2-checkpoint.json'}


def inventory():
    return sorted([str(p.relative_to(REPO)) for p in ROOT.rglob('*')
                   if p.is_file() and p != ROOT/'freeze.json' and str(p.relative_to(ROOT)) not in OUTPUTS
                   and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts] + EXTERNAL)


def verify():
    freeze = json.loads((ROOT/'freeze.json').read_text())
    assert sorted(freeze['sha256']) == inventory(), 'Frozen input inventory changed'
    for name, digest in freeze['sha256'].items():
        assert hashlib.sha256((REPO/name).read_bytes()).hexdigest() == digest, name
    return freeze


def seal():
    if (ROOT/'freeze.json').exists():
        raise SystemExit('Refuse to overwrite freeze')
    inputs = inventory()
    data = dict(schema_version=3, phase='PHASE_1_ONLY_NO_SCORING',
                commit_required_before_phase2=True,
                sha256={n: hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in inputs},
                note='freeze.json cannot hash itself; its hash is anchored by the orchestrator git commit')
    (ROOT/'freeze.json').write_text(json.dumps(data, indent=2)+'\n')
    verify()

if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--seal']:
        seal()
    elif sys.argv[1:] == ['--verify']:
        verify()
        print('Freeze verified')
    else:
        raise SystemExit('Use --seal once, or --verify')

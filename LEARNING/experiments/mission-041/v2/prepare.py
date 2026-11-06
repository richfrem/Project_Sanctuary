"""Generate suites without importing mutation recipes; freeze before curation."""
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import specifications as spec

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
RUNNER = REPO / 'plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/counterexample_eval.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (ROOT / 'freeze.json').exists():
        raise SystemExit('Already frozen; refusing to overwrite.')
    operators = ['unsigned_width_2', 'identifier_casefold', 'stable_sequence_sort',
                 'sequence_tail_loss', 'signedness_erasure', 'identifier_prefix_encoding']
    targets = ['import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nimport references\nimport mutations\n']
    for name in spec.NAMES:
        targets.append(f'def ref_{name}(c):\n    return references.{name}(c)\n')
        targets.append(f'def dev_{name}(c):\n    return None\n')
        for op in operators:
            targets.append(f'def {name}_{op}(c):\n    return mutations.evaluate({name!r}, {op!r}, c)\n')
        if name in spec.OLD:
            existing = f"""import sys
from pathlib import Path
import importlib.util
sys.path.insert(0, str(Path(__file__).resolve().parent))
import specifications
claim_path = Path(__file__).resolve().parent.parent / 'fixtures' / 'claim_{name}.py'
spec = importlib.util.spec_from_file_location('old_claim', claim_path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
CASES = iter(specifications.domain({name!r}))
def generate(rng):
    return next(CASES)
def example_generate(rng):
    return next(CASES)
def expected(c):
    return old.expected(c)
def check(c, actual):
    return old.check(c, actual)
"""
            (ROOT / f'existing_{name}.py').write_text(existing)
    (ROOT / 'targets.py').write_text('\n'.join(targets))
    suites = {}
    manifest = {'components': []}
    for name in spec.NAMES:
        start = time.monotonic()
        suites[name] = {'generation_arm_seconds':{}}
        for arm, generate in [('example', spec.example_generate), ('property', spec.property_generate)]:
            rng = random.Random(41052)
            arm_start = time.monotonic()
            cases = [generate(name, rng) for _ in range(64)]
            generation_elapsed = time.monotonic()-arm_start
            if generation_elapsed > 1.0:
                raise RuntimeError('Generation budget exceeded; do not tune on held-out results.')
            suites[name]['generation_arm_seconds'][arm] = generation_elapsed
            suites[name][arm] = cases
        suites[name]['generation_seconds'] = time.monotonic()-start
        wrapper = f'''"""Frozen claim wrapper for {name}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import specifications as spec
NAME = {name!r}
def generate(rng):
    return spec.property_generate(NAME, rng)
def example_generate(rng):
    return spec.example_generate(NAME, rng)
def expected(c):
    return spec.expected(NAME, c)
def check(c, actual):
    return actual == spec.expected(NAME, c)
'''
        (ROOT / f'claim_{name}.py').write_text(wrapper)
        manifest['components'].append({'id':name, 'family':spec.FAMILIES[name],
            'contract':spec.CONTRACTS[name], 'v1_rerun':name in spec.OLD,
            'existing_tests': 'v1 reference-verification domain and unchanged v1 claim' if name in spec.OLD else 'none; NO_TESTS, not synthetic tests',
            'claim':f'claim_{name}.py'})
    (ROOT / 'suites.json').write_text(json.dumps(suites,indent=2)+'\n')
    (ROOT / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files = sorted([p for p in ROOT.iterdir() if p.suffix in ['.py','.md','.json'] and p.name != 'freeze.json'])
    old = ROOT.parent
    files += [old/'verify_references.py',old/'build_fixtures.py',old/'preregistration.md',old/'results.md']
    files += sorted((old/'fixtures').glob('*.py'))
    files += [RUNNER, RUNNER.parent.parent/'tests/test_counterexample_eval.py']
    (ROOT / 'freeze.json').write_text(json.dumps({'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'seed':41052,'trials':64,'generation_seconds_per_arm':1,'generated_search_seconds':2,'existing_search_seconds':30,
        'component_seconds':600, 'sha256':{str(p.relative_to(REPO)):digest(p) for p in files}},indent=2)+'\n')
    print('FROZEN 20 components, 40 suites, 64 cases each; source hashes in v2/freeze.json')


if __name__ == '__main__':
    main()

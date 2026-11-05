"""Independent oracle cross-checks; exhaustive where feasible, sampled replay."""
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    start = time.monotonic()
    refs = load(ROOT / 'fixtures' / 'references.py')
    rng = random.Random(41000)
    domains = {
        'lease': (list(c) for c in itertools.product(range(33), range(33), [False, True])),
        'path': (list(c) for n in range(9) for c in itertools.product(['a','b','c','.','..'], repeat=n)),
        'accounting': (list(c) for c in itertools.product(range(33), repeat=3)),
        'schema': (list(c) for c in itertools.product(['new','active','closed'], ['open','close','reset'])),
        'clamp': ([value, lower, upper] for value in range(-32,33)
                  for lower in range(-32,33) for upper in range(lower,33)),
    }
    # Replay's full stated domain is too large to exhaust. Exhaust a reduced
    # alphabet including duplicates/negative/zero, plus 10k full-domain samples.
    alphabet = list(itertools.product([0,1], [-1,0,1]))
    replay_claim = load(ROOT / 'fixtures' / 'claim_replay.py')
    domains['replay'] = itertools.chain(
        ([list(event) for event in c] for n in range(7)
         for c in itertools.product(alphabet, repeat=n)),
        (replay_claim.example_generate(rng) for _ in range(10000)))
    results = {}
    for name, cases in domains.items():
        claim = load(ROOT / 'fixtures' / f'claim_{name}.py')
        target = getattr(refs, name)
        count = 0
        failures = []
        for case in cases:
            count += 1
            actual = target(case)
            if not claim.check(case, actual):
                failures.append({'case': case, 'actual': actual, 'expected': claim.expected(case)})
                if len(failures) >= 5:
                    break
        results[name] = {'cases': count, 'failures': failures,
                         'coverage': ('reduced exhaustive + 10000 seeded full-domain samples'
                                      if name == 'replay' else 'exhaustive stated domain')}
    output = {'components': results, 'elapsed_seconds': time.monotonic() - start,
              'all_passed': all(not r['failures'] for r in results.values())}
    (ROOT / 'reference-verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    return 0 if output['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

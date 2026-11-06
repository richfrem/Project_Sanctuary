"""Full finite-domain reference verification, equivalence and kill density.

This is curation, NOT arm scoring. It never imports pools or reads suites and
never computes generated-suite detection. Every applicable candidate is retained.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import components
from specifications import NAMES, domain, expected

ROOT = Path(__file__).resolve().parent


def load(item):
    spec = importlib.util.spec_from_file_location(item['id'], ROOT/item['file'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, item['component'])


def main():
    output = ROOT/'curation.json'
    if output.exists():
        raise SystemExit('Refuse to overwrite curation')
    catalog = json.loads((ROOT/'mutants.json').read_text())
    report = {'method': 'exhaustive over entire declared finite domain',
              'syntactic_noops': 0, 'inapplicable_candidates': 0, 'components': []}
    for name in NAMES:
        start = time.monotonic()
        items = [dict(m, distinguishing_cases=0, first_witness=None) for m in catalog if m['component'] == name]
        functions = [load(item) for item in items]
        count = 0
        for c in domain(name):
            count += 1
            oracle = expected(name, c)
            assert getattr(components, name)(c) == oracle, (name, c)
            for item, function in zip(items, functions):
                try:
                    actual = function(c)
                    differs = actual != oracle
                except Exception as exc:
                    actual = {'exception': type(exc).__name__, 'detail': str(exc)}
                    differs = True
                if differs:
                    item['distinguishing_cases'] += 1
                    if item['first_witness'] is None:
                        item['first_witness'] = {'case': c, 'expected': oracle, 'actual': actual}
        for item in items:
            item['domain_size'] = count
            item['kill_density'] = item['distinguishing_cases']/count
            item['classification'] = 'NON_EQUIVALENT' if item['distinguishing_cases'] else 'TRUE_EQUIVALENT'
            item['proof'] = f'Compared all {count} inputs in specifications.domain({name!r}) against independently verified reference/specification'
        entry = dict(component=name, domain_size=count, reference_mismatches=0,
                     curation_seconds=time.monotonic()-start, candidates=items)
        report['components'].append(entry)
        print(name, count, [round(m['kill_density']*100, 4) for m in items], flush=True)
    all_items = [m for c in report['components'] for m in c['candidates']]
    report['summary'] = dict(applicable_candidates=len(all_items),
        true_equivalents=sum(m['classification'] == 'TRUE_EQUIVALENT' for m in all_items),
        non_equivalents=sum(m['classification'] == 'NON_EQUIVALENT' for m in all_items),
        domain_comparisons=sum(c['domain_size']*len(c['candidates']) for c in report['components']))
    output.write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Per-mutant exhaustive kill density', '',
             'Full declared-domain uniform density; not arm detection. Counts include every input, not just the first witness.', '',
             '| Mutant | Operator | Differing / domain | Density | Classification |',
             '|---|---|---:|---:|---|']
    for m in all_items:
        lines.append(f"| {m['id']} | {m['operator']} | {m['distinguishing_cases']} / {m['domain_size']} | {100*m['kill_density']:.6f}% | {m['classification']} |")
    (ROOT/'kill-density.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()

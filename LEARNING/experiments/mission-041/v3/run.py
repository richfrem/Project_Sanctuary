"""PHASE 2 ONLY. Never invoke until orchestrator commits the phase-1 freeze."""
import argparse
import json
import re
import time
from pathlib import Path
from freeze_tools import verify
from harness import isolated

ROOT = Path(__file__).resolve().parent


def summarize(report):
    components = report['components']
    mutants = [m for c in components for m in c['mutants']]
    n = len(mutants)
    arms = ['property', 'example', 'existing']
    results = [v for c in components for v in c['references'].values()]
    results += [m[a] for m in mutants for a in arms]
    infrastructure = [r for r in results if r['status'] in {'ERROR', 'TIMEOUT'}]
    rates = {a: 100*sum(m[a]['status'] == 'COUNTEREXAMPLE' for m in mutants)/n if n else None for a in arms}
    alarms = {a: sum(c['references'][a]['status'] == 'COUNTEREXAMPLE' for c in components) for a in arms}
    maximum = max((c['component_seconds'] for c in components), default=0)
    valid = (len(components) == 20 and n >= 80 and not infrastructure
             and alarms['example'] == alarms['existing'] == 0)
    thresholds = dict(detection_ge_80=n > 0 and rates['property'] >= 80,
                      gain_ge_20pp=n > 0 and rates['property']-rates['example'] >= 20,
                      false_alarm_le_5=len(components) > 0 and 100*alarms['property']/len(components) <= 5,
                      component_runtime_le_600=bool(components) and maximum <= 600)
    return dict(non_equivalents=n, detection_percent=rates, reference_alarms=alarms,
                reference_denominator=len(components), max_component_seconds=maximum,
                infrastructure_failures=len(infrastructure), valid=valid, thresholds=thresholds,
                verdict='INVALID' if not valid else 'THRESHOLDS_MET' if all(thresholds.values()) else 'FALSIFIED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase2', action='store_true', required=True)
    parser.add_argument('--freeze-commit', required=True,
                        help='Orchestrator attestation: git commit containing this exact freeze; not independently verified here')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', args.freeze_commit):
        parser.error('Supply the full committed freeze hash')
    verify()
    output = ROOT/'phase2-results.json'
    checkpoint = ROOT/'phase2-checkpoint.json'
    if output.exists() or checkpoint.exists():
        raise SystemExit('Refuse overwrite/retry: preserve first registered measurement')
    suites = json.loads((ROOT/'suites.json').read_text())
    curation = json.loads((ROOT/'curation.json').read_text())
    preparation = json.loads((ROOT/'preparation-timing.json').read_text())
    preparation_seconds = preparation['whole_suite_generation_seconds']
    report = dict(freeze_commit_attestation=args.freeze_commit, components=[])
    for c in curation['components']:
        name = c['component']
        start = time.monotonic()
        entry = dict(id=name, references={}, mutants=[])
        def run(file, arm):
            remaining = 600-c['curation_seconds']-preparation_seconds-(time.monotonic()-start)
            if remaining <= 0:
                return dict(status='TIMEOUT', note='Aggregate component ceiling')
            return isolated(ROOT/file, name, name, arm, suites[name][arm], min(2., remaining))
        for arm in ['existing', 'example', 'property']:
            entry['references'][arm] = run('components.py', arm)
        for m in c['candidates']:
            if m['classification'] != 'NON_EQUIVALENT':
                continue
            result = dict(id=m['id'])
            for arm in ['existing', 'example', 'property']:
                result[arm] = run(m['file'], arm)
            entry['mutants'].append(result)
        entry['component_seconds'] = time.monotonic()-start+c['curation_seconds']+preparation_seconds
        report['components'].append(entry)
        checkpoint.write_text(json.dumps(report, indent=2)+'\n')
    verify()
    report['summary'] = summarize(report)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))

if __name__ == '__main__':
    main()

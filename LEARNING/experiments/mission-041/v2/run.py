"""Checkpointed, frozen three-arm experiment, preserving all raw observations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import specifications as spec
from curate import check_freeze

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
RUNNER_PATH = REPO/'plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/counterexample_eval.py'
module_spec = importlib.util.spec_from_file_location('runner',RUNNER_PATH)
runner = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(runner)
ARMS = ['existing','example','property']


def verify_lock():
    freeze = check_freeze()
    lock = json.loads((ROOT/'scoring-lock.json').read_text())
    assert freeze == lock['freeze_sha256']
    assert hashlib.sha256((ROOT/'curation.json').read_bytes()).hexdigest() == lock['curation_sha256']
    return lock


def evaluate(name, selector, arm, suites, count, remaining):
    if arm == 'existing' and name not in spec.OLD:
        return {'status':'NO_TESTS','tested':0,'elapsed_seconds':0,
                'reason':'No historical tests exist for this newly introduced component.'}
    if remaining <= 0:
        return {'status':'TIMEOUT','tested':None,'elapsed_seconds':0}
    if arm == 'existing':
        return runner.search(str(ROOT/'targets.py')+':'+selector,
            str(ROOT/f'existing_{name}.py'),seed=41052,trials=count,
            seconds=min(30,remaining),mode='property')
    result = runner.search(str(ROOT/'targets.py')+':'+selector,
        str(ROOT/f'claim_{name}.py'),seed=41052,trials=64,
        seconds=min(2,remaining),mode='witness',cases=suites[name][arm])
    result['arm'] = arm
    result['suite_sha256'] = hashlib.sha256(json.dumps(suites[name][arm],sort_keys=True).encode()).hexdigest()
    return result


def summarize(components):
    mutants = [m for c in components for m in c['mutants']]
    n = len(mutants)
    detections = {a:sum(m[a]['status']=='COUNTEREXAMPLE' for m in mutants) for a in ARMS}
    rates = {a:100*detections[a]/n if n else None for a in ARMS}
    references = {a:{'n':len(components),'alarms':sum(c['references'][a]['status']=='COUNTEREXAMPLE' for c in components),
        'no_tests':sum(c['references'][a]['status']=='NO_TESTS' for c in components)} for a in ARMS}
    all_results = [r for c in components for r in c['references'].values()]+[m[a] for m in mutants for a in ARMS]
    errors = sum(r['status']=='ERROR' for r in all_results)
    timeouts = sum(r['status']=='TIMEOUT' for r in all_results)
    false_alarms = 100*references['property']['alarms']/len(components) if components else None
    delta = rates['property']-rates['example'] if n else None
    maximum = max((c['total_computational_seconds'] for c in components),default=None)
    thresholds = {'detection_ge_80':bool(n) and rates['property']>=80,
        'delta_ge_20pp':bool(n) and delta>=20,
        'false_alarm_le_5':false_alarms is not None and false_alarms<=5,
        'component_runtime_le_600s':maximum is not None and maximum<=600}
    valid = (len(components)==20 and n>0 and errors==0 and timeouts==0 and
        all(c['unresolved']==0 and c['mutants'] for c in components) and
        all(c['references']['example']['status']=='EXHAUSTED' for c in components))
    return {'n_components':len(components),'n_non_equivalent_mutants':n,
        'n_equivalent_excluded':sum(len(c['equivalents']) for c in components),
        'detections':detections,'detection_percent':rates,'delta_pp':delta,
        'reference_results':references,'property_false_alarm_percent':false_alarms,
        'max_component_computational_seconds':maximum,
        'total_computational_seconds':sum(c['total_computational_seconds'] for c in components),
        'errors':errors,'timeouts':timeouts,'valid_measurement':valid,'thresholds':thresholds,
        'recommendation':'cannot decide' if not valid else 'CORROBORATED' if all(thresholds.values()) else 'FALSIFIED'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='primary-results.json')
    parser.add_argument('--development',action='store_true')
    args = parser.parse_args()
    lock = verify_lock()
    output = ROOT/args.output
    if output.exists():
        raise SystemExit('Output exists; use new version rather than overwrite.')
    suites = json.loads((ROOT/'suites.json').read_text())
    curated = json.loads((ROOT/'curation.json').read_text())
    components = []
    report = {'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'lock':lock,'development':args.development,'components':components}
    for item in curated['components']:
        name = item['id']
        start = time.monotonic()
        prior_seconds = item['elapsed_seconds']+suites[name]['generation_seconds']
        def run(selector, arm):
            remaining = 600-prior_seconds-(time.monotonic()-start)
            return evaluate(name,selector,arm,suites,item['verified_cases'],remaining)
        row = {'id':name,'family':spec.FAMILIES[name],
            'references':{arm:run('ref_'+name,arm) for arm in ARMS},
            'mutants':[], 'equivalents':[op for op in item['operators'] if op['classification']=='EQUIVALENT'],
            'unresolved':sum(op['classification']=='UNRESOLVED' for op in item['operators'])}
        operators = [{'operator':'constant_return','classification':'NON_EQUIVALENT'}] if args.development else item['operators']
        for op in operators:
            if op['classification'] != 'NON_EQUIVALENT':
                continue
            selector = 'dev_'+name if args.development else name+'_'+op['operator']
            row['mutants'].append({'id':selector,'operator':op['operator'],
                **{arm:run(selector,arm) for arm in ARMS}})
        row['scoring_seconds'] = time.monotonic()-start
        row['preparation_seconds'] = prior_seconds
        row['total_computational_seconds'] = prior_seconds+row['scoring_seconds']
        components.append(row)
        report['summary'] = summarize(components)
        checkpoint = output.with_suffix('.checkpoint.json')
        checkpoint.write_text(json.dumps(report,indent=2)+'\n')
        print(name,{a:sum(m[a]['status']=='COUNTEREXAMPLE' for m in row['mutants']) for a in ARMS},
              'n='+str(len(row['mutants'])),'seconds='+str(round(row['total_computational_seconds'],3)),flush=True)
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    report['post_run_lock_verified'] = verify_lock()==lock
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()

"""V3 budget accounting and subprocess execution; import never scores anything."""
import copy
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from specifications import expected
from pools import relation

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
GENERIC = REPO/'plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/counterexample_eval.py'
spec = importlib.util.spec_from_file_location('generic_runner', GENERIC)
generic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generic)


def execute(target, name, arm, suite):
    """Count EVERY invocation, including both sides of metamorphic probes.

    Oracle/relation exceptions propagate as infrastructure errors. Only target
    exceptions or witnessed contract rejections count as counterexamples.
    """
    calls = 0
    def call(case):
        nonlocal calls
        calls += 1
        try:
            return target(copy.deepcopy(case)), None
        except Exception as exc:
            return None, {'status': 'COUNTEREXAMPLE', 'reason': 'target_exception',
                          'case': case, 'detail': type(exc).__name__+': '+str(exc), 'calls': calls}
    for probe in suite:
        if arm == 'property':
            a, b, law = probe
            cases = [a, b]
            oracle = [expected(name, c) for c in cases]
        else:
            cases = [probe[0] if arm == 'existing' else probe]
            oracle = [probe[1] if arm == 'existing' else expected(name, cases[0])]
        outputs = []
        for case, wanted in zip(cases, oracle):
            actual, failure = call(case)
            if failure:
                return failure
            if actual != wanted:
                return dict(status='COUNTEREXAMPLE', reason='independent_specification',
                            case=case, actual=actual, expected=wanted, calls=calls)
            outputs.append(actual)
        if arm == 'property' and not relation(law, *outputs):
            return dict(status='COUNTEREXAMPLE', reason='metamorphic_relation',
                        probe=probe, outputs=outputs, calls=calls)
    return dict(status='EXHAUSTED', calls=calls, note='Not proof of correctness')


def isolated(filename, function, name, arm, suite, seconds=2.0):
    start = time.monotonic()
    request = dict(filename=str(filename), function=function, name=name, arm=arm, suite=suite)
    proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=(os.name == 'posix'))
    try:
        stdout, stderr = proc.communicate(json.dumps(request), timeout=seconds)
        result = json.loads(stdout) if proc.returncode == 0 else dict(status='ERROR', detail=stderr[-2000:])
    except subprocess.TimeoutExpired:
        generic.kill_worker(proc)
        proc.communicate()
        result = dict(status='TIMEOUT', note='Not a kill; inconclusive')
    except (ValueError, TypeError) as exc:
        result = dict(status='ERROR', detail=str(exc))
    result.update(elapsed_seconds=time.monotonic()-start, seconds_budget=seconds)
    return result


def worker():
    try:
        req = json.load(sys.stdin)
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
            target = generic.load_function(req['filename']+':'+req['function'])
            result = execute(target, req['name'], req['arm'], req['suite'])
    except Exception as exc:
        result = dict(status='ERROR', detail=type(exc).__name__+': '+str(exc))
    print(json.dumps(result))

if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Worker only. Scoring requires run.py --phase2 after orchestrator commit.')
    worker()

#!/usr/bin/env python3
"""Frozen deterministic validity/primary/secondary evaluator for MISSION-062.

Usage: python3 evaluate.py (only after committing protocol.md and freeze.json).
Key Input Dependencies: fast C binary/source, reference.py, protocol.md, freeze.json;
    Python stdlib, git, original 3600-second/4-GiB resource budget.
Key Functions: fast_tables(), resource_check(), main(). No network or model calls.
"""
import csv
import hashlib
import io
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import reference as ref

ROOT = Path(__file__).resolve().parent
START = time.monotonic()
PRIOR_WALL_RESERVE = 300  # conservative allowance for pre-freeze development compute
DEADLINE = START + 3600 - PRIOR_WALL_RESERVE
MEMORY_LIMIT = 4 * 1024**3


# Read full histograms from the independent executable and verify all subset totals.
def fast_tables(h: int, k: int, qmax: int) -> dict[int, dict[int, int]]:
    """Run bounded C computation and reject malformed/duplicate/missing histogram rows."""
    remaining = DEADLINE - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('compute budget exhausted')
    run = subprocess.run([str(ROOT / 'fast'), str(h), str(k), str(qmax)],
                         capture_output=True, text=True, check=True, timeout=min(60, remaining))
    tables = {q: {} for q in range(k, qmax + 1)}
    for row in csv.DictReader(io.StringIO(run.stdout)):
        q, size, count = (int(row[key]) for key in ('q', 'size', 'count'))
        if q not in tables or size in tables[q] or count <= 0 or not 1 <= size <= math.comb(h+k-1,k-1):
            raise ValueError('malformed histogram row')
        tables[q][size] = count
    for q, histogram in tables.items():
        if sum(histogram.values()) != math.comb(q, k):
            raise ValueError(f'subset total mismatch h={h},k={k},q={q}')
    (ROOT / 'runs' / f'h{h}-k{k}-q{qmax}.csv').write_text(run.stdout)
    resource_check()
    return tables


# Check observed process maxima in platform-correct units and the full compute clock.
def resource_check(enforce: bool = True) -> dict:
    """Return observed resource evidence, refusing a breached original ceiling."""
    factor = 1 if sys.platform == 'darwin' else 1024
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    maxrss = factor * (own.ru_maxrss + children.ru_maxrss)
    elapsed = time.monotonic() - START
    if enforce and (elapsed + PRIOR_WALL_RESERVE > 3600 or maxrss > MEMORY_LIMIT):
        raise TimeoutError('original resource ceiling exceeded')
    return {'elapsed_s': elapsed, 'prior_compute_wall_upper_bound_s': PRIOR_WALL_RESERVE, 'conservative_peak_rss_bytes': maxrss,
            'cpu_s': own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime,
            'rss_method': 'sum of parent and child maximum RSS (conservative, not simultaneous sampling)'}


# Gate correctness before using the instrument for the primary registered hypothesis.
def main() -> int:
    """Verify committed hashes, validate, evaluate, and retain a precise finite verdict."""
    frozen = json.loads((ROOT / 'freeze.json').read_text())
    for filename, expected in frozen['sha256'].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen artifact changed: '+filename)
    for filename in ('protocol.md', 'freeze.json'):
        subprocess.run(['git', 'ls-files', '--error-unmatch', str(ROOT / filename)],
                       cwd=ROOT, capture_output=True, check=True)
        if subprocess.run(['git','diff','--quiet','HEAD','--',str(ROOT/filename)],cwd=ROOT).returncode:
            raise ValueError('freeze is not committed cleanly')
    (ROOT / 'runs').mkdir(exist_ok=True)
    result = {'assessment': 'UNASSESSED', 'validity': {}, 'primary': {}, 'secondary': {}}
    try:
        matches = 0
        for k in (3, 4):
            for h in range(2, 7):
                fast = fast_tables(h,k,40)
                slow = ref.histograms(h,k,40, deadline=DEADLINE)
                if fast != slow:
                    raise ValueError(f'independent histogram mismatch h={h}, k={k}')
                matches += len(fast)
                resource_check()
        calibration = []
        for h,k,expected,limit in ((3,3,5,120),(4,3,6,120),(2,4,8,60),(3,4,16,50)):
            tables = fast_tables(h,k,limit)
            actual = ref.threshold(tables,h,k)
            if actual != expected or ref.ok(tables[expected-1],h,k):
                raise ValueError(f'published threshold mismatch {h},{k}: {actual}')
            calibration.append({'h':h,'k':k,'threshold':actual,'verified_through':limit,'preceding_cell_fails':True})
        if any(len(ref.sumset((1,2,3),h)) != 2*h+1 for h in range(2,65)):
            raise ValueError('arithmetic progression identity failed')
        result['validity'] = {'passed':True,'independent_histogram_cells':matches,
                              'scope':'k=3,4; h=2..6; every q=k..40','published_cells':calibration,
                              'arithmetic_progression_checked_h':[2,64]}
        cells = 0
        primary_tables = {}
        for h in range(5,17):
            tables = fast_tables(h,3,250)
            primary_tables[h] = tables
            for q in range(3*h,251):
                cells += 1
                if not ref.ok(tables[q],h,3):
                    independent = ref.histograms(h,3,q,deadline=DEADLINE)[q]
                    if independent != tables[q]:
                        raise ValueError('counterexample did not reproduce independently')
                    result['assessment'] = 'FALSIFIED'
                    result['primary'] = {'covered_cells':cells,'counterexample':{'h':h,'q':q,'histogram':tables[q]}}
                    return 0
        result['assessment'] = 'CORROBORATED'
        result['primary'] = {'covered_cells':cells,'expected_cells':sum(251-3*h for h in range(5,17)),
                             'all_passed':True,'scope':'k=3,h=5..16,q=3h..250'}
        thresholds = []
        for h in range(2,17):
            tables = primary_tables.get(h) or fast_tables(h,3,250)
            q0 = ref.threshold(tables,h,3)
            thresholds.append({'h':h,'finite_threshold':q0,'ratio':q0/h if q0 else None,
                               'proof_sufficient_bound':8*math.factorial(3)*h*(2*h+1)**4+1})
        xs,ys = [math.log(row['h']) for row in thresholds],[math.log(row['finite_threshold']) for row in thresholds]
        xbar,ybar = sum(xs)/len(xs),sum(ys)/len(ys)
        slope = sum((x-xbar)*(y-ybar) for x,y in zip(xs,ys))/sum((x-xbar)**2 for x in xs)
        result['secondary'] = {'k3':thresholds,'descriptive_log_log_slope':slope,
                               'k4':[{'h':h,'finite_threshold_through_80':ref.threshold(fast_tables(h,4,80),h,4)} for h in (2,3,4)]}
        return 0
    except (ValueError, TimeoutError, subprocess.SubprocessError) as error:
        result['assessment'] = 'UNASSESSED'
        result['error'] = str(error)
        return 2
    finally:
        result['resources'] = resource_check(enforce=False)
        result['freeze_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        (ROOT / 'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__ == '__main__':
    raise SystemExit(main())

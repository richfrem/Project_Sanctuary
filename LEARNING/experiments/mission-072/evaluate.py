#!/usr/bin/env python3
"""Frozen exact finite restart experiment for MISSION-072.

Usage: python3 evaluate.py, after committing protocol/freeze/instruments.
Key Input Dependencies: discounted.py, direct.py, development.json, frozen protocol;
    Fraction, gzip, git and Python standard library. No models/network during execution.
Key Functions: resources(), check_time(), row_value(), dense_gate(), main().
"""
import csv
from fractions import Fraction as F
import gzip
import hashlib
import io
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import direct
import discounted as fast

ROOT = Path(__file__).resolve().parent
BEGAN = time.monotonic()
PRIOR_DEVELOPMENT_S = json.loads((ROOT/'development.json').read_text())['elapsed_s']
RATES = sorted({F(1,2**j) for j in range(1,25)} | {F(3,4),F(7,8)})


# Observed memory is an upper bound formed from process maxima, not an invented live sample.
def resources() -> dict:
    """Return measured elapsed/CPU/max-RSS with platform-correct units."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    factor = 1 if sys.platform == 'darwin' else 1024
    return {'elapsed_s':time.monotonic()-BEGAN,'prior_development_s':PRIOR_DEVELOPMENT_S,'cpu_s':own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime,
            'conservative_peak_rss_bytes':factor*(own.ru_maxrss+child.ru_maxrss)}


# Fail rather than truncate the declared domain when the original resource ceiling is reached.
def check_time() -> None:
    """Enforce the frozen 900-second, one-GiB compute ceiling."""
    state = resources()
    if state['elapsed_s'] + PRIOR_DEVELOPMENT_S > 900 or state['conservative_peak_rss_bytes'] > 1024**3:
        raise TimeoutError('frozen resource budget exhausted')


# Rational text is canonical evidence; float conversion is limited to descriptive summaries.
def row_value(value: F) -> str:
    """Serialize an exact rational as numerator/denominator, including integers."""
    return f'{value.numerator}/{value.denominator}'


# Direct full-matrix solves validate an independently different representation on held-out points.
def dense_gate() -> dict:
    """Compare 48 prespecified reset means and baseline moments exactly, no tolerance."""
    count = 0
    for length in (9,16,24,32):
        mean,second = fast.moments(length)
        assert (mean,second) == direct.baseline(length)
        assert mean == [F(x*(2*length-x)) for x in range(1,length+1)]
        for r in (F(1,2),F(1,16)):
            u = fast.survival_sum(length,r)
            for start in (1,(length+1)//2,length):
                for cost in (F(0),F(2)):
                    actual = direct.reset_time(length,start,r,cost)
                    if actual != fast.restart_mean(u[start-1],r,cost):
                        raise ValueError('independent held-out mean mismatch')
                    count += 1
                    check_time()
    return {'passed':True,'held_out_dense_comparisons':count,'baseline_moments_exact':True}


# Complete the finite classification test without changing the registered domain after observation.
def main() -> int:
    """Verify the committed freeze, validate, test each case and preserve raw exact means."""
    freeze = json.loads((ROOT/'freeze.json').read_text())
    for name,digest in freeze['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen artifact changed: '+name)
    for name in ('protocol.md','freeze.json'):
        subprocess.run(['git','ls-files','--error-unmatch',str(ROOT/name)],cwd=ROOT,capture_output=True,check=True)
        if subprocess.run(['git','diff','--quiet','HEAD','--',str(ROOT/name)],cwd=ROOT).returncode:
            raise ValueError('protocol/freeze not committed cleanly')
    (ROOT/'raw').mkdir(exist_ok=True)
    result = {'assessment':'UNASSESSED','validity':{},'primary':{},'cases':[]}
    case_count = comparisons = improving = 0
    try:
        result['validity'] = dense_gate()
        for length in range(9,33):
            means,seconds = fast.moments(length)
            u = {r:fast.survival_sum(length,r) for r in RATES}
            with (ROOT/'raw'/f'L{length:02}.csv.gz').open('wb') as binary:
                with gzip.GzipFile(fileobj=binary,mode='wb',mtime=0) as zipped:
                    with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as output:
                        writer = csv.writer(output,lineterminator='\n')
                        writer.writerow(['L','start','cost','r','exact_mean','strictly_beneficial'])
                        for start in range(1,length+1):
                            mean,second = means[start-1],seconds[start-1]
                            for cost in fast.costs(mean,second):
                                check_time()
                                prediction = fast.local_predicts(mean,second,cost)
                                values = [(r,fast.restart_mean(u[r][start-1],r,cost)) for r in RATES]
                                actual = any(value < mean for _,value in values)
                                best_r,best_mean = min(values,key=lambda pair:pair[1])
                                for r,value in values:
                                    writer.writerow([length,start,row_value(cost),row_value(r),row_value(value),int(value<mean)])
                                comparisons += len(values)
                                case_count += 1
                                improving += int(actual)
                                case = {'L':length,'start':start,'cost':row_value(cost),'moment_prediction':prediction,
                                        'observed_grid_benefit':actual,'baseline_mean':row_value(mean),
                                        'best_grid_r':row_value(best_r),'best_grid_mean':row_value(best_mean),
                                        'best_grid_ratio':float(best_mean/mean)}
                                result['cases'].append(case)
                                if actual != prediction:
                                    direct_values = [(r,direct.reset_time(length,start,r,cost)) for r in RATES]
                                    if direct_values != values:
                                        raise ValueError('candidate counterexample failed independent mean reproduction')
                                    result['assessment'] = 'FALSIFIED'
                                    result['primary'] = {'covered_cases':case_count,'rate_comparisons':comparisons,
                                                         'improving_cases':improving,'counterexample':case,
                                                         'independent_reproduction':True}
                                    return 0
        # Reconstruct the exact inventory from the frozen state/cost rule, not an observed success count.
        expected = sum(len(fast.costs(mu,s)) for length in range(9,33)
                       for mu,s in zip(*fast.moments(length)))
        if case_count != expected or comparisons != expected*len(RATES):
            raise ValueError('registered coverage inventory mismatch')
        result['assessment'] = 'CORROBORATED'
        result['primary'] = {'covered_cases':case_count,'expected_cases':expected,'rate_comparisons':comparisons,
                             'rates_per_case':len(RATES),'improving_cases':improving,'all_classifications_match':True,
                             'scope':'L9..32,all starts,frozen exact cost set and probability grid only'}
        return 0
    except (ValueError,ArithmeticError,TimeoutError,AssertionError) as error:
        result['assessment'] = 'UNASSESSED'
        result['error'] = str(error)
        result['partial_coverage'] = {'cases':case_count,'rate_comparisons':comparisons}
        return 2
    finally:
        result['resources'] = resources()
        result['freeze_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        summary = {key:value for key,value in result.items() if key != 'cases'}
        print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    raise SystemExit(main())

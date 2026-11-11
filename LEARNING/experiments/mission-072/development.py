#!/usr/bin/env python3
"""Development-only instrument validation; never evaluates L>=9.

Usage: python3 development.py
Key Input Dependencies: direct.py, discounted.py, Fraction; no network/models.
Key Functions: main().
"""
from fractions import Fraction as F
import json
from pathlib import Path
import time

import direct
import discounted as fast


# Compare independent methods on a development domain fixed before any held-out result.
def main() -> None:
    """Check baseline identities, clock convention and 216 exact reset means."""
    began = time.monotonic()
    comparisons = 0
    for length in range(1,9):
        mean,second = fast.moments(length)
        slow_mean,slow_second = direct.baseline(length)
        assert (mean,second) == (slow_mean,slow_second)
        assert mean == [F(x*(2*length-x)) for x in range(1,length+1)]
        for r in (F(1,2),F(1,8)):
            u = fast.survival_sum(length,r)
            for start in range(1,length+1):
                for cost in (F(0),F(1),F(2)):
                    assert fast.restart_mean(u[start-1],r,cost) == direct.reset_time(length,start,r,cost)
                    comparisons += 1
    # One-step deterministic completion: free resets cost nothing; timed resets delay completion.
    for r in (F(1,2),F(1,8)):
        assert direct.reset_time(1,1,r,F(0)) == 1
        assert direct.reset_time(1,1,r,F(1)) == 1/(1-r)
    result = {'passed':True,'lengths':[1,8],'reset_mean_comparisons':comparisons,
              'baseline_moments_exact':True,'mu_equals_x_times_2L_minus_x':True,
              'one_step_clock_checks':True,'elapsed_s':time.monotonic()-began,
              'held_out_evaluated':False}
    Path(__file__).with_name('development.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

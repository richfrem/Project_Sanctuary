#!/usr/bin/env python3
"""Post-hoc limitation example, excluded from MISSION-072's primary evidence.

Usage: python3 posthoc_illustration.py
Key Input Dependencies: Python Fraction; independent attempt durations1/100 with probabilities1/10,9/10.
Key Functions: main(). Exact renewal PGF is cross-checked with a separate state recurrence.
"""
from fractions import Fraction as F
import json
from pathlib import Path


# Construct an exact counterexample to an untested universal local-to-global generalization.
def main() -> None:
    """Verify a two-duration process in two representations without changing the mission verdict."""
    mu,second = F(901,10),F(90001,10)
    r,z = F(9,10),F(1,10)
    pgf = F(1,10)*z+F(9,10)*z**100
    mean = z*(1-pgf)/(r*pgf)
    derivative = mu*mu-(second-mu)/2-mu
    # Progress states need i remaining ordinary steps; B_i=a_i+b_i*M_start.
    a,b = F(0),F(0)
    for _ in range(99):
        a,b = z*(1+a),z*b+r
    state_mean = z*(1+F(9,10)*a)/(1-r-z*F(9,10)*b)
    if state_mean != mean or not (derivative >= 0 and mean < mu):
        raise ValueError('post-hoc example did not verify')
    result = {'status':'post-hoc illustration only; outside registered reflected-walk primary domain',
              'process':'Independent attempt duration1 with probability1/10,100 with probability9/10; instantaneous reset, fresh independent attempt each restart.',
              'r':str(r),'baseline_mean':str(mu),'derivative_at_zero':str(derivative),
              'mean_at_r':str(mean),'mean_at_r_decimal':float(mean),'strictly_improves':True,
              'independent_state_recurrence_matches':True,
              'interpretation':'A nonnegative local derivative does not generally rule out improvement at a distant restart rate. This is not evidence against the finite reflected-walk result and not a novelty claim.'}
    Path(__file__).with_name('posthoc-illustration.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Verified post-hoc example in two representations: mean90.1 -> approximately11 although local derivative is positive.')


if __name__ == '__main__':
    main()

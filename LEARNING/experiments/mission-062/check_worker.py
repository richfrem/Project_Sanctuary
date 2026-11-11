#!/usr/bin/env python3
"""Check the unmodified worker proposal against its original development contract.

Usage: python3 check_worker.py worker-03-proposal.c
Key Input Dependencies: C compiler, Python standard library; no held-out inputs.
Key Functions: main(). Tests literal subset totals on h=2,k=3,q<=8.
"""
import csv
import io
import math
import subprocess
import sys
import tempfile
from pathlib import Path


# Validate the worker's own output, not the supervisor-corrected final instrument.
def main() -> int:
    """Return failure if compile or exact development histogram totals disagree."""
    with tempfile.TemporaryDirectory() as directory:
        binary = str(Path(directory)/'worker')
        subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror',sys.argv[1],'-o',binary],check=True,timeout=30)
        result = subprocess.run([binary,'2','3','8'],capture_output=True,text=True,check=True,timeout=10)
        totals = {}
        for row in csv.DictReader(io.StringIO(result.stdout)):
            q = int(row['q'])
            totals[q] = totals.get(q,0)+int(row['count'])
        expected = {q:math.comb(q,3) for q in range(3,9)}
        if totals != expected:
            print('FAIL: worker counts normalized shapes, not all translated subsets:',totals,'expected',expected)
            return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

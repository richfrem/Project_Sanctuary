"""Verify references and curate every equivalent BEFORE scoring any arm."""
import hashlib
import json
from pathlib import Path
import sys
import time
import specifications as spec
import references
import mutations

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def check_freeze():
    frozen = json.loads((ROOT/'freeze.json').read_text())
    for rel, expected in frozen['sha256'].items():
        if hashlib.sha256((REPO/rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen source changed: '+rel)
    return hashlib.sha256((ROOT/'freeze.json').read_bytes()).hexdigest()


def main():
    check_freeze()
    if (ROOT/'curation.json').exists():
        raise SystemExit('Curation exists; preserve it.')
    all_rows = []
    for name in spec.NAMES:
        start = time.monotonic()
        candidates = {op: {'operator':op,'witness':None,'input_changes':0} for op in mutations.OPERATORS}
        n = 0
        target = getattr(references,name)
        for c in spec.domain(name):
            n += 1
            expected = spec.expected(name,c)
            actual = target(c)
            if actual != expected:
                raise AssertionError(('reference disagrees',name,c,expected,actual))
            for op, row in candidates.items():
                # Need only one distinguishing witness; identity/equivalence
                # claims still scan the entire verification domain.
                if row['witness'] is not None:
                    continue
                transformed = mutations.transform(name,op,c)
                row['input_changes'] += transformed != c
                mutated = target(transformed)
                if mutated != expected:
                    row['witness'] = {'case':c,'reference':actual,'mutant':mutated}
        for op,row in candidates.items():
            if row['witness'] is not None:
                row.update(classification='NON_EQUIVALENT', proof='Explicit valid-domain distinguishing witness; independent oracle accepts reference.')
            elif name != 'replay':
                row.update(classification='EQUIVALENT',proof='Exhaustive equality over entire declared finite input domain.')
            elif op in ['identifier_casefold','identifier_prefix_encoding']:
                row.update(classification='EQUIVALENT',proof='Structural identity: replay inputs contain only lists and integers; string-only operator cannot alter any full-domain input.')
            else:
                row.update(classification='UNRESOLVED',proof='No witness in reduced+sampled domain is not a proof of full-domain equivalence.')
        entry = {'id':name,'verified_cases':n,'verification_scope':
            'reduced exhaustive (IDs 0,1; values -1,0,1; length <=6) plus 10000 full-domain seeded samples' if name == 'replay' else 'exhaustive declared finite domain',
            'operators':list(candidates.values()),'elapsed_seconds':time.monotonic()-start}
        all_rows.append(entry)
        print(name,n, {kind:sum(r['classification']==kind for r in candidates.values()) for kind in ['NON_EQUIVALENT','EQUIVALENT','UNRESOLVED']},flush=True)
        (ROOT/'curation-checkpoint.json').write_text(json.dumps(all_rows,indent=2)+'\n')
    report = {'freeze_sha256':check_freeze(),'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'components':all_rows}
    (ROOT/'curation.json').write_text(json.dumps(report,indent=2)+'\n')
    lock = {'curation_sha256':hashlib.sha256((ROOT/'curation.json').read_bytes()).hexdigest(),'freeze_sha256':check_freeze()}
    (ROOT/'scoring-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print('CURATION LOCKED before scoring')


if __name__ == '__main__':
    main()

"""Postmeasurement verification of frozen evidence; never selects or tunes tests."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import specifications
import references
import mutations
from curate import check_freeze


def main():
    start = time.monotonic()
    freeze = check_freeze()
    lock = json.loads((ROOT/'scoring-lock.json').read_text())
    assert freeze == lock['freeze_sha256']
    assert hashlib.sha256((ROOT/'curation.json').read_bytes()).hexdigest() == lock['curation_sha256']
    curated = json.loads((ROOT/'curation.json').read_text())
    report = json.loads((ROOT/'primary-results.json').read_text())
    suites = json.loads((ROOT/'suites.json').read_text())
    assert len(report['components']) == 20
    kills = {a:0 for a in ['existing','example','property']}
    replayed = {a:0 for a in kills}
    equivalents = non_equivalents = existing_covered = unresolved = 0
    survivor_rows = []
    for c, prep in zip(report['components'],curated['components']):
        name = c['id']
        assert name == prep['id']
        operators = {o['operator']:o for o in prep['operators']}
        assert len(operators)==6
        expected_ids = {o['operator'] for o in prep['operators'] if o['classification']=='NON_EQUIVALENT'}
        assert {m['operator'] for m in c['mutants']} == expected_ids
        equivalents += sum(o['classification']=='EQUIVALENT' for o in prep['operators'])
        unresolved += sum(o['classification']=='UNRESOLVED' for o in prep['operators'])
        non_equivalents += len(c['mutants'])
        ref = getattr(references,name)
        for m in c['mutants']:
            op = m['operator']
            witness = operators[op]['witness']
            assert ref(witness['case']) == specifications.expected(name,witness['case'])
            assert mutations.evaluate(name,op,witness['case']) != specifications.expected(name,witness['case'])
            existing_covered += name in specifications.OLD
            for arm in kills:
                result = m[arm]
                if result['status']=='NO_TESTS':
                    assert arm=='existing' and name not in specifications.OLD and result['tested']==0
                elif result['status']=='COUNTEREXAMPLE':
                    case = result['case']
                    actual = mutations.evaluate(name,op,case)
                    assert actual != specifications.expected(name,case)
                    assert ref(case)==specifications.expected(name,case)
                    assert result['reason']=='property_rejected'
                    assert actual==result['actual']
                    if arm != 'existing':
                        assert result['case']==suites[name][arm][result['tested']-1]
                    kills[arm] += 1
                    replayed[arm] += 1
                else:
                    assert result['status']=='EXHAUSTED'
                    if arm != 'existing':
                        assert result['tested']==64
                        # Replay the exact frozen suite only, not new examples.
                        assert all(mutations.evaluate(name,op,x)==specifications.expected(name,x) for x in suites[name][arm])
                    if arm=='property':
                        survivor_rows.append({'component':name,'operator':op,'tested':result['tested'],
                            'curation_witness':witness,'baseline_counterexample':m['example'].get('case')})
        for arm in ['example','property']:
            assert len(suites[name][arm])==64
            assert suites[name]['generation_arm_seconds'][arm] <= 1
            assert c['references'][arm]['status']=='EXHAUSTED'
            assert c['references'][arm]['tested']==64
    assert unresolved==0
    assert equivalents+non_equivalents==120
    assert kills==report['summary']['detections']
    assert all(c['total_computational_seconds']<=600 for c in report['components'])
    result = {'audit_kind':'post-run evidence replay, no new test selection or scored search',
        'freeze_verified':True,'curation_lock_verified':True,'n_components':20,
        'n_verified_reference_cases':sum(c['verified_cases'] for c in curated['components']),
        'n_candidate_mutants':120,'n_equivalents':equivalents,'n_non_equivalents':non_equivalents,
        'n_unresolved':unresolved,'detections_recomputed':kills,'detection_witnesses_replayed':replayed,
        'existing_covered_mutants':existing_covered,'existing_uncovered_mutants':non_equivalents-existing_covered,
        'existing_detection_percent_among_covered':100*kills['existing']/existing_covered,
        'property_survivors':survivor_rows,
        'zero_of_20_alarm_one_sided_95pct_upper_percent':100*(1-0.05**(1/20)),
        'measured_audit_seconds':time.monotonic()-start,
        'all_passed':True,'primary_sha256':hashlib.sha256((ROOT/'primary-results.json').read_bytes()).hexdigest()}
    output = ROOT/'audit.json'
    if output.exists():
        raise SystemExit('Audit output exists; preserve it.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

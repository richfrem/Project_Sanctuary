"""Offline tests for real-agent admission and independent semantic checking."""
from pathlib import Path
import importlib.util
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import real_agent as ra
import pytest


def test_sequential_operations_have_hand_computed_answers():
    t=dict(initial='09',operands=[29]*100)
    assert ra.expected(t,0,'09')=='38'
    assert ra.expected(t,1,'38')=='83'
    assert ra.expected(t,2,'83')=='38'
    assert ra.expected(t,3,'99')=='28'
    ra.verify(t,[dict(index=0,answer='38'),dict(index=1,answer='83'),dict(index=2,answer='38')])
    with pytest.raises(AssertionError): ra.verify(t,[dict(index=0,answer='39')])
    with pytest.raises(AssertionError): ra.verify(t,[dict(index=1,answer='38')])


def test_fixtures_disjoint_and_all_event_types_frozen():
    f=ra.fixtures()
    assert all(len(v)==30 for v in f.values())
    assert {x['task']['seed'] for x in f['development']}.isdisjoint({x['task']['seed'] for x in f['held_out']})
    for split in ('development','held_out'):
        for fixture in f[split]:
            assert len(fixture['task']['operands'])==100
            assert {e['kind'] for e in fixture['events']}==set(ra.KINDS)
            assert sorted(e['token_boundary'] for e in fixture['events'])==[e['token_boundary'] for e in fixture['events']]
    assert all(not x['events'] for x in f['control'])
    assert [x['task'] for x in f['control']]==[x['task'] for x in f['held_out']]


def test_prompt_bounds_fit_budget_and_checkpoint_is_not_free():
    for fixture in ra.fixtures()['held_out']:
        t=fixture['task']
        for i in range(100):
            work=ra.work_messages(t,i,'99','A')
            cp=ra.checkpoint_messages(i,'99')
            assert 0<ra.bound(work)+ra.bound(cp)<ra.BUDGET
            assert all(m['role'] in ('system','user') for m in work+cp)


def test_best_fixed_excludes_adaptive_and_tie_breaks_by_checkpoint_cost():
    summaries={'adaptive-v2':dict(verified_subgoals=999,checkpoint_tokens=0),
        'end-only':dict(verified_subgoals=20,checkpoint_tokens=50),
        'fixed-steps-1':dict(verified_subgoals=20,checkpoint_tokens=60)}
    assert ra.best_fixed(summaries)=='end-only'

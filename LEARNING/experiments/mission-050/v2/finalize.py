#!/usr/bin/env python3
"""Create accounting and handoff from completed, audited evidence only."""
import collections
import contextlib
import io
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import real_agent as ra
import report_results


def lines(name):
    p=HERE/name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        report_results.render()
    trials=ra.load_trials(); calls=lines('real-calls.jsonl'); events=lines('real-events.jsonl')
    summary=json.loads((HERE/'real-summary.json').read_text())
    accounting=dict(
        completed_real_trials=len(trials),planned_real_trials=630,
        measured_prompt_tokens=sum(c['usage']['prompt_tokens'] for c in calls),
        measured_completion_tokens=sum(c['usage']['completion_tokens'] for c in calls),
        measured_total_tokens=sum(c['usage']['total_tokens'] for c in calls),
        measured_cached_prompt_tokens=sum(c['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0) for c in calls),
        calls=len(calls),work_calls=sum(c['op']=='work' for c in calls),
        wrong_work_calls=sum(c['op']=='work' and not c['correct'] for c in calls),
        checkpoint_calls=sum(c['op']=='checkpoint' for c in calls),
        role_B_work_calls=sum(c['op']=='work' and 'Agent role: B.' in c['request']['messages'][0]['content'] for c in calls),
        invalid_checkpoint_responses=sum(c['op']=='checkpoint' and not c['correct'] for c in calls),
        lock_wait_seconds=sum(c['lock_wait_seconds'] for c in calls),
        request_wall_seconds=sum(c['wall_seconds'] for c in calls),
        observed_events=dict(collections.Counter(e['event']['kind'] for e in events if e['type']=='interruption')),
        simulated_denials=sum(e['type']=='denied_request' for e in events),
        http_or_usage_errors=[e for e in events if e['type']=='request_error'],
        adaptive_nonfinal_requests=sum(o['requested'] and not o['final_flush'] for t in trials if t['policy']=='adaptive-v2' for o in t['observations']),
        experimental_token_ceiling=1000000,
        supervisor_session_tokens='unavailable; never self-reported',
        per_split={})
    starts=[e['utc'] for e in events if e['type']=='trial_start']
    accounting['wall_seconds_since_real_start']=time.time()-min(starts) if starts else None
    resume_path=HERE/'resume-round.json'
    if resume_path.exists():
        resume=json.loads(resume_path.read_text())
        added=calls[resume['baseline_calls']:]
        accounting['resumed_round']=dict(baseline_completed_trials=resume['baseline_completed'],
            additional_completed_trials=len(trials)-resume['baseline_completed'],
            additional_calls=len(added),
            measured_prompt_tokens=sum(c['usage']['prompt_tokens'] for c in added),
            measured_completion_tokens=sum(c['usage']['completion_tokens'] for c in added),
            measured_total_tokens=sum(c['usage']['total_tokens'] for c in added),
            lock_wait_seconds=sum(c['lock_wait_seconds'] for c in added),
            request_wall_seconds=sum(c['wall_seconds'] for c in added))
        completed={t['trial'] for t in trials}
        remaining=[t for t in resume['authorized_remaining_trials'] if t['trial'] not in completed]
        ra.write_json(HERE/'remaining.json',dict(remaining_trials=len(remaining),
            counts_by_split=dict(collections.Counter(t['split'] for t in remaining)),
            next_trial=remaining[0]['trial'] if remaining else None,remaining=remaining))
    for split in ('development','held_out','control'):
        accounting['per_split'][split]={}
        for policy in ra.POLICIES:
            subset=[t for t in trials if t['split']==split and t['policy']==policy]
            if not subset: continue
            a=ra.aggregate(subset)
            ids={t['trial'] for t in subset}
            a['observed_event_kinds']=dict(collections.Counter(e['event']['kind'] for e in events if e['type']=='interruption' and e['trial'] in ids))
            a['scheduled_events']=sum(t['scheduled_interruptions'] for t in subset)
            a['observed_events']=sum(t['observed_interruptions'] for t in subset)
            a['quota_denials']=sum(t['quota_denials'] for t in subset)
            a['nonfinal_adaptive_requests']=sum(o['requested'] and not o['final_flush'] for t in subset for o in t['observations'])
            a['allocated_checkpoint_overhead']=a['checkpoint_tokens']/a['budget']
            a['consumed_budget_fraction']=a['measured_tokens']/a['budget']
            accounting['per_split'][split][policy]=a
    ra.write_json(HERE/'final-accounting.json',accounting)
    # Provisional rubric assessment, not a budget/contribution ledger write.
    assessment=dict(scope='worker experiment deliverable; provisional assessor codex-b',
        dimensions=dict(useful_outcome=2 if len(trials)==630 else 1,learning_depth=2,
            verification_rigor=2,token_economy=2,anti_asch_integrity=3),
        prediction_ref='design.md',measurement_ref='final-accounting.json',
        outcome='Frozen v2 simulation missed gain, loss and control throughput. Real finite benchmark: '+json.loads((HERE/'verdict.json').read_text())['real_candidate'],
        interpretation='Negative evidence is a useful outcome. Early lock contention consumed time; calls were metered. No held-out tuning. Verification rating is 2 because no independent agent review was authorized.',
        ledger_status='not written: user explicitly reserves ledgers/manifests for orchestrator')
    ra.write_json(HERE/'investment-review.json',assessment)
    frozen=json.loads((HERE/'freeze.json').read_text())
    for path,h in frozen['hashes'].items(): assert hashlib.sha256((ra.ROOT/path).read_bytes()).hexdigest()==h,path
    for name,h in frozen['v1_hashes'].items(): assert hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==h,name
    files=[p for p in HERE.rglob('*') if p.is_file() and p.name not in ('seal.json','finalize-output.json') and not any(part.startswith('pytest-') or part=='__pycache__' for part in p.relative_to(HERE).parts)]
    seal=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    ra.write_json(HERE/'seal.json',seal)
    print(json.dumps({k:v for k,v in accounting.items() if k!='per_split'},indent=2))

if __name__=='__main__': main()

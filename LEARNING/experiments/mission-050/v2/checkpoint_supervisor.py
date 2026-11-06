#!/usr/bin/env python3
"""Operational deadline guard; never changes experiment parameters or results.

Use the runner's existing snapshot-exists guard to stop BEFORE the next trial.
Create only an explicitly marked stop file; remove it after the runner exits.
No in-flight request is interrupted. Completed-trial resume is supported by the
frozen runner. This wrapper adds no model calls and no scoring changes.
"""
import argparse
import datetime
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stop-at-utc',required=True)
    args=parser.parse_args()
    deadline=datetime.datetime.fromisoformat(args.stop_at_utc).timestamp()
    while time.time()<deadline:
        if 'COMPLETE ' in (HERE/'real-run.log').read_text(): return
        time.sleep(min(20,max(0,deadline-time.time())))
    import real_agent as ra
    fs=json.loads((HERE/'real-fixtures.json').read_text())
    ordered=[]
    for split in ('development','held_out','control'):
        for i,f in enumerate(fs[split]):
            policies=ra.POLICIES[i%len(ra.POLICIES):]+ra.POLICIES[:i%len(ra.POLICIES)]
            ordered.extend(f"{split}-{f['task']['seed']}-{p}" for p in policies)
    events=[json.loads(l) for l in (HERE/'real-events.jsonl').read_text().splitlines()]
    starts=[e['trial'] for e in events if e['type']=='trial_start']
    current=starts[-1]; index=ordered.index(current)
    if index+1==len(ordered): return
    next_trial=ordered[index+1]
    marker=HERE/'snapshots'/f'{next_trial}.json'
    guard=dict(operational_stop_marker=True,reason='checkpoint round deadline; stop before starting another measured trial',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),current_trial=current,next_trial=next_trial)
    with marker.open('x') as out: json.dump(guard,out)
    (HERE/'checkpoint-stop.json').write_text(json.dumps(guard,indent=2)+'\n')
    print('DEADLINE GUARD ARMED '+next_trial,flush=True)

if __name__=='__main__': main()

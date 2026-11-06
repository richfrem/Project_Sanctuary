#!/usr/bin/env python3
"""Run the preregistered v2 simulation without modifying v1 artifacts."""
import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts'))
import adaptive_checkpoint as ac


def run():
    started = time.perf_counter()
    policies = (*ac.fixed_family(), ac.AdaptivePolicyV2())
    fixtures = {}
    for split, task_seed, schedule_seed in [('development',110000,120000),('held_out',130000,140000)]:
        fixtures[split] = [dict(task=dataclasses.asdict(task := ac.make_task(task_seed+i)),
            events=[dataclasses.asdict(e) for e in ac.make_schedule(schedule_seed+i,task.budget,i%6)],
            profile=i%6,schedule_seed=schedule_seed+i) for i in range(30)]
    fixtures['control'] = [dict(f,events=[],schedule_seed=None) for f in fixtures['held_out']]
    splits = {}
    selected = None
    for split, fs in fixtures.items():
        trials = []
        for f in fs:
            t = f['task']
            task = ac.Task(t['seed'],tuple(ac.Subgoal(**g) for g in t['goals']),t['checkpoint_cost'],t['budget'])
            for p in policies:
                trials.append(ac.run_trial(task,tuple(ac.Event(**e) for e in f['events']),p,
                    trace=isinstance(p,ac.AdaptivePolicyV2)))
        summaries = {p.name: ac.summarize([t for t in trials if t['policy']==p.name]) for p in policies}
        fixed = [p.name for p in ac.fixed_family()]
        winner = min(fixed,key=lambda n:(-summaries[n]['verified_subgoals'],-summaries[n]['verified_value'],summaries[n]['checkpoint_ticks'],n))
        if selected is None:
            selected = winner
            (Path(__file__).parent/'simulation-selected-fixed.json').write_text(json.dumps({'selected':selected})+'\n')
        splits[split] = dict(trials=trials,summaries=summaries,best_fixed=winner,
            vs_development_selected=ac.compare(summaries['adaptive-v2'],summaries[selected]),
            vs_split_best=ac.compare(summaries['adaptive-v2'],summaries[winner]))
    control=splits['control']
    control['throughput_loss']=1-control['summaries']['adaptive-v2']['verified_subgoals']/control['summaries'][control['best_fixed']]['verified_subgoals']
    result=dict(version='mission-050-v2',units='synthetic ticks, not inference tokens',
        prior=[1,40],fixtures=fixtures,fixture_sha256=ac.digest_json(fixtures),
        selected_fixed=selected,splits=splits)
    out=Path(__file__).parent
    raw=json.dumps(result,sort_keys=True,separators=(',',':')).encode()
    (out/'simulation-evidence.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    summary={k:v for k,v in result.items() if k not in ('fixtures','splits')}
    summary['splits']={k:{x:y for x,y in v.items() if x!='trials'} for k,v in splits.items()}
    (out/'simulation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    execution=dict(wall_seconds=time.perf_counter()-started,evidence_sha256=hashlib.sha256(raw).hexdigest())
    (out/'simulation-execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    print(json.dumps(execution))

if __name__=='__main__': run()

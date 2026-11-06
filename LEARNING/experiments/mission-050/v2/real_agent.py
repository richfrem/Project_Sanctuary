#!/usr/bin/env python3
"""Preregistered local Qwen experiment. All server requests use exclusive flock.

Stdlib only. Evidence is append-only. Worker processes are disposable; model
server ownership remains with the human. Read design.md before running.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts'))
import adaptive_checkpoint as ac

MODEL='mlx-community/Qwen2.5-Coder-7B-Instruct-4bit'
ENDPOINT='http://127.0.0.1:8080/v1'
BUDGET=1000
POLICIES=('end-only','fixed-steps-1','fixed-steps-2','fixed-steps-4','fixed-time-240','fixed-time-480','adaptive-v2')
KINDS=('context_reset','worker_termination','quota_denial','vendor_handoff')
WORK_SYSTEM='Follow the instruction. Reply with exactly two digits and nothing else.'
CHECKPOINT_SYSTEM='Copy the requested checkpoint exactly. No explanation.'


def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'))
def digest(obj): return hashlib.sha256(canonical(obj).encode()).hexdigest()
def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def append(path,obj):
    with path.open('a') as f:
        f.write(canonical(obj)+'\n'); f.flush(); os.fsync(f.fileno())


def task(seed):
    rng=random.Random(seed)
    return dict(seed=seed,initial=f'{rng.randrange(100):02d}',operands=[rng.randrange(1,40) for _ in range(100)])


def expected(t,index,state):
    if index%3==0: return f"{(int(state)+t['operands'][index])%100:02d}"
    if index%3==1: return state[::-1]
    return ''.join(sorted(state))


def work_messages(t,index,state,role):
    op=(f"Add {t['operands'][index]} modulo 100." if index%3==0 else
        'Reverse the two digits.' if index%3==1 else 'Sort the two digits in ascending order.')
    return [dict(role='system',content=WORK_SYSTEM+' Agent role: '+role+'.'),
        dict(role='user',content=f'State: {state}. {op} Write the resulting two-digit state, padding with a leading zero when necessary.')]


def checkpoint_messages(cursor,state):
    obj=dict(cursor=cursor,state=state)
    return [dict(role='system',content=CHECKPOINT_SYSTEM),
        dict(role='user',content='Return exactly this JSON: '+canonical(obj))]


def bound(messages): return sum(len(m['content'].encode()) for m in messages)+128+24


def request(messages):
    body=dict(model=MODEL,messages=messages,temperature=0,max_tokens=24,stream=False)
    started=time.monotonic()
    with open('/tmp/sanctuary-mlx.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        acquired=time.monotonic()
        req=urllib.request.Request(ENDPOINT+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as response:
            raw=json.load(response)
    usage=raw.get('usage',{})
    if any(not isinstance(usage.get(k),int) or usage[k]<0 for k in ('prompt_tokens','completion_tokens','total_tokens')):
        raise RuntimeError('Missing/noninteger server usage: '+canonical(raw))
    if usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']:
        raise RuntimeError('Inconsistent server token accounting')
    return dict(request=body,response=raw,usage=usage,lock_wait_seconds=acquired-started,wall_seconds=time.monotonic()-started)


def agent(conn,t,receipts,role):
    """Agent holds volatile progress. Parent independently checks every answer."""
    cursor=len(receipts); state=receipts[-1]['answer'] if receipts else t['initial']
    while True:
        command=conn.recv()
        if command['op']=='stop': break
        if command['op']=='restore':
            receipts=command['receipts']; cursor=len(receipts)
            state=receipts[-1]['answer'] if receipts else t['initial']
            conn.send(dict(cursor=cursor,state=state)); continue
        try:
            messages=(checkpoint_messages(cursor,state) if command['op']=='checkpoint' else work_messages(t,cursor,state,role))
            result=request(messages)
            content=result['response']['choices'][0]['message']['content'].strip()
            result.update(cursor=cursor,state_before=state,content=content,pid=os.getpid())
            if command['op']=='work':
                correct=bool(re.fullmatch(r'\d{2}',content)) and content==expected(t,cursor,state)
                result['correct']=correct
                if correct:
                    receipts=receipts+[dict(index=cursor,answer=content)]
                    cursor+=1; state=content
            else:
                try: result['correct']=json.loads(content)==dict(cursor=cursor,state=state)
                except (ValueError,TypeError): result['correct']=False
            conn.send(result)
        except Exception as e:
            conn.send(dict(error=type(e).__name__+': '+str(e))); break
    conn.close()


def start_agent(t,receipts,role):
    parent,child=mp.Pipe()
    proc=mp.Process(target=agent,args=(child,t,receipts,role))
    proc.start(); child.close()
    return proc,parent


def stop_agent(proc,conn,kill=False):
    if kill: proc.terminate()
    else:
        try: conn.send(dict(op='stop'))
        except (BrokenPipeError,OSError): pass
    proc.join(timeout=5)
    if proc.is_alive(): proc.terminate(); proc.join()
    conn.close()


def verify(t,receipts):
    state=t['initial']
    for i,r in enumerate(receipts):
        assert r['index']==i and r['answer']==expected(t,i,state)
        state=r['answer']
    return state


def fixtures():
    data={}
    for split,ts,ss in [('development',210000,220000),('held_out',230000,240000)]:
        fs=[]
        for i in range(30):
            rng=random.Random(ss+i)
            events=[dict(token_boundary=b+rng.randint(-30,30),kind=KINDS[(i+j)%4]) for j,b in enumerate((120,270,420,570))]
            fs.append(dict(task=task(ts+i),schedule_seed=ss+i,events=events))
        data[split]=fs
    data['control']=[dict(f,events=[],schedule_seed=None) for f in data['held_out']]
    return data


def run_trial(split,fixture,policy):
    t=fixture['task']; ident=f"{split}-{t['seed']}-{policy}"
    snapshot=HERE/'snapshots'/f'{ident}.json'
    if snapshot.exists(): raise RuntimeError('Partial trial snapshot exists: '+ident)
    calls_path=HERE/'real-calls.jsonl'; audit_path=HERE/'real-events.jsonl'
    durable=[]; volatile=[]; spent=cp_tokens=work_tokens=acks=lost=interruptions=0
    commits=attempts=call_count=denials=0; last_checkpoint=0; cp_costs=[]; cp_pending=[]
    role='A'; deny_next=False; event_index=0; observations=[]; overruns=[]
    proc,conn=start_agent(t,durable,role)
    started=time.monotonic()
    write_json(snapshot,dict(receipts=[],initial=True))
    try:
        while len(volatile)<100:
            state=verify(t,volatile)
            wm=work_messages(t,len(volatile),state,role); cm=checkpoint_messages(len(volatile),state)
            pending=len(volatile)>len(durable)
            final=pending and BUDGET-spent<=bound(wm)+bound(cm)
            obs=ac.AdaptiveObservation((interruptions+1)/(spent+400),len(volatile)-len(durable),
                acks/work_tokens if work_tokens else 0.0, math.ceil(sum(cp_costs)/len(cp_costs)) if cp_costs else 100)
            if policy=='adaptive-v2':
                requested=ac.AdaptivePolicyV2().should_checkpoint(obs)
                observations.append(dict(spent=spent,observation=vars_for_slots(obs),requested=requested,final_flush=final))
            elif policy.startswith('fixed-steps-'): requested=len(volatile)-len(durable)>=int(policy.rsplit('-',1)[1])
            elif policy.startswith('fixed-time-'): requested=spent-last_checkpoint>=int(policy.rsplit('-',1)[1])
            else: requested=False
            op='checkpoint' if pending and (final or requested) else 'work'
            messages=cm if op=='checkpoint' else wm
            if BUDGET-spent<bound(messages): break
            if deny_next:
                append(audit_path,dict(trial=ident,type='denied_request',op=op,spent=spent,server_tokens=0))
                denials+=1; deny_next=False; continue
            conn.send(dict(op=op))
            result=conn.recv()
            if 'error' in result:
                append(audit_path,dict(trial=ident,type='request_error',result=result,spent=spent))
                raise RuntimeError(result['error'])
            call_count+=1
            usage=result['usage']; tokens=usage['total_tokens']; spent+=tokens
            if tokens>bound(messages) or spent>BUDGET: overruns.append(dict(call=call_count,tokens=tokens,spent=spent,bound=bound(messages)))
            append(calls_path,dict(trial=ident,split=split,policy=policy,seed=t['seed'],call=call_count,op=op,spent=spent,**result))
            if result['request']['messages']!=messages: raise AssertionError('Agent observation mismatch')
            assert result['cursor']==len(volatile) and result['state_before']==state
            if op=='work':
                work_tokens+=tokens
                correct=bool(re.fullmatch(r'\d{2}',result['content'])) and result['content']==expected(t,len(volatile),state)
                assert correct==result['correct']
                if correct:
                    volatile.append(dict(index=len(volatile),answer=result['content'])); acks+=1
            else:
                cp_tokens+=tokens; cp_costs.append(tokens); attempts+=1
            crossed=[]
            while event_index<len(fixture['events']) and fixture['events'][event_index]['token_boundary']<=spent:
                event=fixture['events'][event_index]; event_index+=1; interruptions+=1; crossed.append(event)
                discarded=len(volatile)-len(durable); lost+=discarded
                disk=json.loads(snapshot.read_text())['receipts']; verify(t,disk); assert disk==durable
                volatile=[dict(r) for r in disk]
                before_pid=proc.pid
                if event['kind'] in ('worker_termination','vendor_handoff'):
                    stop_agent(proc,conn,kill=event['kind']=='worker_termination')
                    if event['kind']=='vendor_handoff': role='B' if role=='A' else 'A'
                    proc,conn=start_agent(t,disk,role)
                else:
                    conn.send(dict(op='restore',receipts=disk)); restored=conn.recv()
                    assert restored==dict(cursor=len(disk),state=verify(t,disk))
                    if event['kind']=='quota_denial': deny_next=True
                append(audit_path,dict(trial=ident,type='interruption',event=event,spent=spent,lost=discarded,
                    old_pid=before_pid,new_pid=proc.pid,durable=len(durable)))
            if op=='checkpoint' and not crossed and result['correct']:
                verify(t,volatile)
                candidate=dict(receipts=volatile,sha256=digest(volatile),tokens=spent)
                temp=snapshot.with_suffix('.tmp'); write_json(temp,candidate); os.replace(temp,snapshot)
                durable=[dict(r) for r in volatile]; last_checkpoint=spent; commits+=1
            if overruns or final: break
        lost+=len(volatile)-len(durable)
        assert acks==len(durable)+lost
        verify(t,durable)
        return dict(trial=ident,split=split,policy=policy,seed=t['seed'],budget=BUDGET,
            measured_tokens=spent,prompt_completion_scope='server usage prompt_tokens + completion_tokens',
            work_tokens=work_tokens,checkpoint_tokens=cp_tokens,unused_budget=BUDGET-spent,
            verified_subgoals=len(durable),acknowledged_subgoals=acks,lost_acknowledged_subgoals=lost,
            loss_fraction=lost/acks if acks else 0.0,overhead_fraction=cp_tokens/spent if spent else None,
            checkpoint_attempts=attempts,checkpoint_commits=commits,observed_interruptions=interruptions,
            scheduled_interruptions=len(fixture['events']),quota_denials=denials,calls=call_count,
            receipts=durable,receipts_sha256=digest(durable),observations=observations,
            wall_seconds=time.monotonic()-started,overruns=overruns)
    finally: stop_agent(proc,conn)


def vars_for_slots(obj): return {n:getattr(obj,n) for n in obj.__slots__}


def aggregate(trials):
    sums={k:sum(t[k] for t in trials) for k in ('budget','measured_tokens','work_tokens','checkpoint_tokens',
        'unused_budget','verified_subgoals','acknowledged_subgoals','lost_acknowledged_subgoals','calls','observed_interruptions')}
    sums.update(n=len(trials),loss_fraction=sums['lost_acknowledged_subgoals']/sums['acknowledged_subgoals'] if sums['acknowledged_subgoals'] else 0,
        overhead_fraction=sums['checkpoint_tokens']/sums['measured_tokens'] if sums['measured_tokens'] else None,
        worst_task_loss=max(t['loss_fraction'] for t in trials),
        worst_task_overhead=max(t['overhead_fraction'] for t in trials))
    return sums


def load_trials():
    path=HERE/'real-trials.jsonl'
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def best_fixed(summaries):
    return min((p for p in summaries if p!='adaptive-v2'),key=lambda n:(-summaries[n]['verified_subgoals'],summaries[n]['checkpoint_tokens'],n))


def summarize():
    trials=load_trials(); selection_path=HERE/'real-selected-fixed.json'
    selected=json.loads(selection_path.read_text())['selected'] if selection_path.exists() else None
    result=dict(trials_completed=len(trials),expected_trials=630,selected_fixed=selected,splits={})
    for split in ('development','held_out','control'):
        subset=[t for t in trials if t['split']==split]
        if not subset: continue
        summaries={p:aggregate([t for t in subset if t['policy']==p]) for p in POLICIES if any(t['policy']==p for t in subset)}
        data=dict(summaries=summaries,complete=len(subset)==210)
        if data['complete']:
            best=best_fixed(summaries); a=summaries['adaptive-v2']; b=summaries[best]
            data.update(best_fixed=best,gain_vs_split_best=a['verified_subgoals']/b['verified_subgoals']-1 if b['verified_subgoals'] else None,
                gain_vs_dev_selected=a['verified_subgoals']/summaries[selected]['verified_subgoals']-1 if selected and summaries[selected]['verified_subgoals'] else None,
                improvement_pass=a['verified_subgoals']*100>=b['verified_subgoals']*120 and b['verified_subgoals']>0,
                loss_pass=a['loss_fraction']<=.05,overhead_pass=a['overhead_fraction']<=.10)
            if split=='control':
                data['throughput_loss']=1-a['verified_subgoals']/b['verified_subgoals'] if b['verified_subgoals'] else None
                data['throughput_pass']=data['throughput_loss'] is not None and data['throughput_loss']<=.03
        result['splits'][split]=data
    result['measured_server_tokens']=sum(t['measured_tokens'] for t in trials)
    result['all_thresholds_pass']=len(trials)==630 and all(result['splits'][s][g] for s in ('development','held_out') for g in ('improvement_pass','loss_pass','overhead_pass')) and result['splits'].get('control',{}).get('throughput_pass',False)
    write_json(HERE/'real-summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('freeze','run','summarize'))
    args=parser.parse_args()
    if args.mode=='freeze':
        if (HERE/'freeze.json').exists(): raise RuntimeError('Already frozen')
        write_json(HERE/'real-fixtures.json',fixtures())
        paths=[HERE/'design.md',HERE/'real_agent.py',HERE/'run_simulation.py',HERE/'real-fixtures.json',
            ROOT/'plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/adaptive_checkpoint.py']
        write_json(HERE/'freeze.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            v1_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.parent.iterdir() if p.is_file()}))
        print('FROZEN: designs, sources, all real fixtures, v1 artifact hashes'); return
    if args.mode=='summarize':
        print(canonical(summarize())); return
    frozen=json.loads((HERE/'freeze.json').read_text())
    for path,h in frozen['hashes'].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    fs=json.loads((HERE/'real-fixtures.json').read_text())
    (HERE/'snapshots').mkdir(exist_ok=True)
    trials=load_trials(); done={t['trial'] for t in trials}
    for split in ('development','held_out','control'):
        for i,f in enumerate(fs[split]):
            # Rotate execution order across tasks to distribute shared-server drift.
            policies=POLICIES[i%len(POLICIES):]+POLICIES[:i%len(POLICIES)]
            for policy in policies:
                ident=f"{split}-{f['task']['seed']}-{policy}"
                if ident in done: continue
                append(HERE/'real-events.jsonl',dict(trial=ident,type='trial_start',utc=time.time()))
                trial=run_trial(split,f,policy)
                append(HERE/'real-trials.jsonl',trial); trials.append(trial)
                print(canonical(dict(trial=ident,verified=trial['verified_subgoals'],tokens=trial['measured_tokens'],trials_completed=len(trials))),flush=True)
                if trial['overruns']: raise RuntimeError('Token admission overrun')
        if split=='development':
            summaries={p:aggregate([t for t in trials if t['split']==split and t['policy']==p]) for p in POLICIES}
            selected=best_fixed(summaries)
            path=HERE/'real-selected-fixed.json'
            selection=dict(selected=selected,development_summaries=summaries)
            if path.exists(): assert json.loads(path.read_text())==selection
            else: write_json(path,selection)
        summarize()
    print('COMPLETE '+canonical(dict(measured_server_tokens=sum(t['measured_tokens'] for t in trials),trials=len(trials))),flush=True)

if __name__=='__main__': main()

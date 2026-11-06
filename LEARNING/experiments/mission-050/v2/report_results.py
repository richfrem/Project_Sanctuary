#!/usr/bin/env python3
"""Render descriptive results; missing real splits remain explicitly undecidable."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import real_agent as ra


def pct(x): return 'undefined' if x is None else f'{x:.4%}'


def render():
    s=json.loads((HERE/'simulation-summary.json').read_text())
    r=ra.summarize()
    lines=['# MISSION-050 v2 results','',
        'Frozen candidate; no held-out iteration. V1 files and results preserved.',
        'See `design.md`, `freeze.json` and `audit.json` for parameters and verification.','',
        '## Simulation — complete','',
        '30 tasks per split; 18 policies; 1,620 trials; 1,200 synthetic ticks per trial.',
        'Same workload distribution as v1, fresh v2 seeds. Prior: 1 interruption / 40 ticks.',
        'Best fixed selected on development: `'+s['selected_fixed']+'`.','',
        '| Split (n=30 adaptive) | Verified adaptive / split-best fixed | Gain vs dev-selected / split-best (>=20%) | Lost acknowledged count / value (<=5%) | Synthetic overhead (<=10%) |',
        '|---|---:|---:|---:|---:|']
    for split,d in s['splits'].items():
        a=d['summaries']['adaptive-v2']; b=d['summaries'][d['best_fixed']]
        lines.append(f"| {split} | {a['verified_subgoals']} / {b['verified_subgoals']} (`{d['best_fixed']}`) | {pct(d['vs_development_selected']['verified_subgoal_gain'])} / {pct(d['vs_split_best']['verified_subgoal_gain'])} | {a['lost_acknowledged_subgoals']}/{a['acknowledged_subgoals']} = {pct(a['loss_count_fraction'])} / {pct(a['loss_value_fraction'])} | {a['checkpoint_ticks']}/{a['budget']} = {pct(a['overhead_fraction'])} |")
    lines+=['',f"Control throughput loss: **{pct(s['splits']['control']['throughput_loss'])}**, n=30 paired tasks, versus best fixed end-only; threshold <=3%: MISS.",
        '', 'Candidate verdict in the simulation: **FALSIFIED**. Development and held-out',
        'miss completion gain and acknowledged-loss gates; controls miss throughput.',
        'Synthetic overhead passes its tick proxy, but **cannot decide inference-token overhead**.',
        'The simulator supports claims about its exact rollback/resource model only.','',
        '## Real-agent arm','',f"Completed {r['trials_completed']}/630 planned trials; measured usage in completed trials: {r['measured_server_tokens']} server tokens.",
        '30 tasks per split, seven policies; 100 available sequential subgoals per task;',
        '1,000 allocated inference tokens per trial. Score durable, independently verified',
        'subgoals; loss denominator includes every acknowledged successful redo.',
        'All checkpoint requests, including failed/interrupted ones, count in measured overhead.',
        'Allocated budgets are equal; unused tokens remain charged for throughput comparisons.','',
        '| Split | Adaptive n | Verified adaptive / best fixed | Gain vs dev-selected / split-best (>=20%) | Ack loss (<=5%) | Measured checkpoint overhead (<=10%) |',
        '|---|---:|---:|---:|---:|---:|']
    for split in ('development','held_out','control'):
        d=r['splits'].get(split)
        if not d or 'adaptive-v2' not in d['summaries']:
            lines.append(f'| {split} | 0 | not measured | cannot decide | cannot decide | cannot decide |'); continue
        a=d['summaries']['adaptive-v2']
        if d['complete']:
            b=d['summaries'][d['best_fixed']]
            verified=f"{a['verified_subgoals']} / {b['verified_subgoals']} (`{d['best_fixed']}`)"
            gain=pct(d['gain_vs_dev_selected'])+' / '+pct(d['gain_vs_split_best'])
        else: verified=str(a['verified_subgoals'])+' / incomplete'; gain='cannot decide: split incomplete'
        lines.append(f"| {split} | {a['n']} | {verified} | {gain} | {a['lost_acknowledged_subgoals']}/{a['acknowledged_subgoals']} = {pct(a['loss_fraction'])} | {a['checkpoint_tokens']}/{a['measured_tokens']} = {pct(a['overhead_fraction'])} |")
    lines+=['']
    control=r['splits'].get('control',{})
    if control.get('complete'):
        a=control['summaries']['adaptive-v2']; b=control['summaries'][control['best_fixed']]
        selected=control['summaries'][r['selected_fixed']]
        selected_loss=1-a['verified_subgoals']/selected['verified_subgoals'] if selected['verified_subgoals'] else None
        lines.append(f"Real interruption-free throughput loss: **{pct(control['throughput_loss'])}**, {a['verified_subgoals']} adaptive / {b['verified_subgoals']} strongest tested fixed durable subgoals, n=30 matched tasks, threshold <=3%: **{'PASS' if control['throughput_pass'] else 'MISS'}**.")
        lines.append(f"Against the frozen development-selected `{r['selected_fixed']}` control comparator: {a['verified_subgoals']} / {selected['verified_subgoals']}, throughput loss {pct(selected_loss)}. The primary control gate uses the strongest tested fixed control.")
    else: lines.append('Real interruption-free throughput gate: **cannot decide**, control split incomplete.')
    completed_misses=[]
    for split in ('development','held_out'):
        d=r['splits'].get(split,{})
        if d.get('complete'):
            completed_misses.extend(split+': '+gate for gate in ('improvement_pass','loss_pass','overhead_pass') if not d[gate])
    if control.get('complete') and not control['throughput_pass']:
        completed_misses.append('control: throughput_pass')
    verdict='FALSIFIED' if completed_misses else 'CORROBORATED' if r['all_thresholds_pass'] else 'cannot decide'
    ra.write_json(HERE/'verdict.json',dict(simulation='FALSIFIED',real_candidate=verdict,
        complete_real_experiment=r['trials_completed']==630,completed_gate_misses=completed_misses,
        reason=('Any completed n=30 primary gate miss falsifies this candidate. All 630 planned trials are complete; verdict scoped to the frozen v2 candidate and 1,000-token design.' if r['trials_completed']==630 else 'Any completed n=30 primary gate miss falsifies this candidate; unfinished conditions remain unmeasured. Full corroboration requires every planned condition.')))
    lines.append('Real local-driver candidate verdict: **'+verdict+'** on completed n=30 primary gates. A single completed primary-gate miss is sufficient under the manifest.')
    if r['trials_completed']!=630:
        lines.append('The full experiment is **unfinished**. Missing split/threshold results remain **cannot decide**; partial descriptive ratios do not replace the frozen experiment.')
    trials=ra.load_trials()
    lines += ['', '### Completed matched task blocks (descriptive only)', '',
        '| Split | Fully matched tasks | Adaptive / strongest fixed verified | Gain | Ack loss | Token overhead |',
        '|---|---:|---:|---:|---:|---:|']
    partial_cohorts={}
    for split in ('development','held_out','control'):
        split_trials=[t for t in trials if t['split']==split]
        seeds={t['seed'] for t in split_trials}
        complete_seeds={seed for seed in seeds if {t['policy'] for t in split_trials if t['seed']==seed}==set(ra.POLICIES)}
        if not complete_seeds:
            lines.append(f'| {split} | 0 | unavailable | cannot decide | unavailable | unavailable |')
            continue
        summaries={policy:ra.aggregate([t for t in split_trials if t['seed'] in complete_seeds and t['policy']==policy]) for policy in ra.POLICIES}
        best=ra.best_fixed(summaries); a=summaries['adaptive-v2']; b=summaries[best]
        gain=a['verified_subgoals']/b['verified_subgoals']-1 if b['verified_subgoals'] else None
        partial_cohorts[split]=dict(task_seeds=sorted(complete_seeds),summaries=summaries,best_fixed_retrospective=best,gain=gain)
        lines.append(f"| {split} | {len(complete_seeds)} | {a['verified_subgoals']} / {b['verified_subgoals']} (`{best}`) | {pct(gain)} | {a['lost_acknowledged_subgoals']}/{a['acknowledged_subgoals']} = {pct(a['loss_fraction'])} | {a['checkpoint_tokens']}/{a['measured_tokens']} = {pct(a['overhead_fraction'])} |")
    ra.write_json(HERE/'real-matched-cohorts.json',partial_cohorts)
    lines += ['', 'These matched-block comparisons use the retrospective strongest fixed within',
        'each displayed cohort. The primary table also reports the frozen development-selected',
        'comparator. An incomplete cohort cannot establish the planned n=30 held-out gates.']
    early=sum(o['requested'] and not o['final_flush'] for t in trials if t['policy']=='adaptive-v2' for o in t['observations'])
    lines += ['', '### Adaptive policy interpretation', '',
        f'The real adaptive policy made **{early} early checkpoint requests** under the frozen design.',
        'With zero early requests it degenerated to end-only: any checkpoint was a final flush.',
        'This is a **valid negative finding for this frozen design**, not an implementation defect',
        'in translating the registered policy. The independent audit checks each decision against',
        'the frozen formula. The design couples a 100-token initial checkpoint-cost prior with',
        'a nine-cost overhead floor, requiring at least 900 work tokens before a first early',
        'request; conservative admission and final flushing stop work earlier in these short',
        '1,000-token trials. This is a design feasibility limitation. It does not test an',
        'adaptive scheduler operating with learned checkpoint costs or meaningful early decisions.',
        'No policy, prior, budget, threshold, prompt or seed was tuned after seeing these results.', '',
        '### Claims supported by each arm', '',
        '| Hypothesis claim | Simulation | Real-agent arm |',
        '|---|---|---|',
        '| >=20% durable completion gain and <=5% acknowledged loss | Tests these claims within the frozen synthetic rollback/value model; misses refute this candidate there. | Completed n=30 interrupted splits test these claims for this local Qwen driver; misses refute this candidate there. |',
        '| <=10% checkpoint inference-token overhead | Cannot support it: tick overhead is only a proxy. | Tests pooled server-reported prompt+completion checkpoint overhead, including failed calls; pooled compliance does not establish worst-task compliance. |',
        '| <=3% interruption-free throughput loss | Tests the synthetic control claim; 3.5998% misses. | Completed n=30 real controls test the local-driver claim at equal allocated token budgets; incomplete controls cannot decide it. |',
        '| Adaptive early scheduling improves real agent recovery | Synthetic scheduling decisions can be examined, but cannot establish real model recovery. | Cannot support this mechanism claim: zero early requests reduce adaptive to end-only. |',
        '| General production, cross-vendor, live-generation or real quota-outage recovery | Cannot support these claims. | Cannot support these claims: safe-point driver recovery only; handoff and quota denial were simulated. |',
        '', 'Neither arm establishes universal optimality over every fixed interval or a population',
        'effect beyond its finite tasks. Fresh v2 seeds also prevent a paired causal claim that',
        'the cold-start change itself caused any difference from v1.']
    lines+=['','The real arm tests local Qwen semantic subgoals, actual measured inference cost,',
        'safe-point context rollback, disposable worker process termination and atomic disk',
        'snapshot recovery. Quota denial is local injection. Vendor handoff is simulated',
        'between agent roles using the same model, so **actual cross-vendor claims remain',
        'undecidable**. No live generation/server is terminated; no production crash durability',
        'or wall-clock fixed-time performance claim follows. Admission byte bounds are',
        'conservative guards, not token telemetry. No population confidence claim is made.','',
        '## Prediction review and integration','',
        'The frozen prediction that cold-start repair would not resolve loss/completion',
        'tradeoffs was borne out in simulation. Simulation control throughput also missed.',
        'The real control result is reported separately above. Server usage',
        'and lock waits are empirical measurements; supervisor session tokens are not reported.',
        'Worker scope prohibits budget/scoring ledgers, manifests, APM install or flight-board',
        'writes. Orchestrator must record verdicts/ROI and perform installation if integrating.',
        'Control-plane status reported INCONSISTENT (missing DB/hooks); bootstrap was GREEN.',
        'No lifecycle repair outside allowed paths was attempted.','']
    disclosures=HERE/'reviewer-disclosures.md'
    if disclosures.exists() and r['trials_completed']==630:
        lines.append(disclosures.read_text())
    (HERE/'results.md').write_text('\n'.join(lines))
    print(f'Rendered results: {r["trials_completed"]}/630 real trials')

if __name__=='__main__': render()

#!/usr/bin/env python3
"""Selection for claim B (prospective matched disagreement cohorts). Deterministic given pool.json, formula.json and declarations/*.json.

Policy F  : top K=16 by the existing priority formula l*d/(c*f), tie -> id.
Policy P  : formula ranks 1..11  +  5 interest-led slots.
Interest-led slots: candidates the formula would NOT select (formula rank > 16), ranked by the rank-sum of the agents' percentile ranks over the four
dimensions (an agent's ratings of its own proposals are excluded; originality is capped when the candidate duplicates the knowledge base).
Cohorts : P-only = the 5 interest picks; F-only = formula ranks 12..16 (the items P gave up). Ranks 1..11 are shared and are NOT executed.
"""
import json
import pathlib
import random
import sys

from pool import STUDY, T, containment, kb_docs


def fscore(s):
    return s["strategic_leverage"] * s["epistemic_depth"] / (s["cost_weight"] * s["operational_friction"])


def percentile_ranks(values):
    """values: {id: rating}. Returns {id: percentile in [0,1], 1 = best}; ties share the mid-rank."""
    n = len(values)
    out = {}
    for k, v in values.items():
        below = sum(1 for x in values.values() if x < v)
        equal = sum(1 for x in values.values() if x == v)
        out[k] = (below + 0.5 * (equal - 1)) / (n - 1) if n > 1 else 0.5
    return out


def main():
    pool = json.load(open(STUDY / "pool.json"))
    cards = {c["id"]: c for c in pool}
    formula = json.load(open(STUDY / "formula.json"))
    decl = {a: {r["id"]: r for r in json.load(open(STUDY / "declarations" / f"{a}.json"))} for a in T["declarers"]}
    proposer = {c["id"]: c.get("proposed_by") for c in pool}

    docs = kb_docs()
    contain = {}
    for cid, c in cards.items():
        contain[cid] = containment(c, {k: v for k, v in docs.items() if k != cid})
    capped = {cid for cid, v in contain.items() if v >= T["originality"]["containment_threshold"]}

    ranked = sorted(cards, key=lambda cid: (-fscore(formula[cid]), cid))
    frank = {cid: i + 1 for i, cid in enumerate(ranked)}

    # interest rank-sum over eligible candidates (formula rank > K)
    elig = [cid for cid in ranked if frank[cid] > T["slots_K"]]
    score = {cid: 0.0 for cid in elig}
    raw = {cid: [] for cid in elig}
    for a, rows in decl.items():
        for dim in T["dimensions"]:
            vals = {}
            for cid in elig:
                if proposer.get(cid) == a or cid not in rows:
                    continue
                v = rows[cid][dim]
                if dim == "originality" and cid in capped:
                    v = min(v, T["originality"]["cap_value"])
                vals[cid] = v
                raw[cid].append(v)
            for cid, p in percentile_ranks(vals).items():
                score[cid] += p
    inter = sorted(elig, key=lambda cid: (-score[cid], -(sum(raw[cid]) / max(1, len(raw[cid]))), cid))

    shared = ranked[:T["formula_share_slots"]]
    f_only = ranked[T["formula_share_slots"]:T["slots_K"]]
    p_only = inter[:T["interest_share_slots"]]
    P = shared + p_only
    F = ranked[:T["slots_K"]]

    # blind codes, execution order, worker assignment (frozen rule)
    rng = random.Random(T["assignment"]["order_seed"])
    combined = [(cid, "P-only") for cid in p_only] + [(cid, "F-only") for cid in f_only]
    rng.shuffle(combined)
    codes = {cid: f"ITEM-{i + 1:02d}" for i, (cid, _) in enumerate(combined)}
    workers = T["assignment"]["workers"]
    assign = {}
    for cohort, start in (("P-only", 0), ("F-only", 1)):
        members = [cid for cid, co in combined if co == cohort]
        for i, cid in enumerate(members):
            assign[cid] = workers[(start + i) % 2]

    sel = dict(
        K=T["slots_K"], formula_ranking=[dict(id=cid, rank=frank[cid], formula=round(fscore(formula[cid]), 2)) for cid in ranked],
        interest_ranking=[dict(id=cid, rank_sum=round(score[cid], 3), mean_raw=round(sum(raw[cid]) / max(1, len(raw[cid])), 2), raters=len(raw[cid]) // 4) for cid in inter],
        originality_capped=sorted(capped), policy_F=F, policy_P=P, shared_not_executed=shared, P_only=p_only, F_only=f_only,
        cohort_sizes=dict(P_only=len(p_only), F_only=len(f_only)),
        cohort_ok=(len(p_only) >= T["cohort_min_items"] and len(f_only) >= T["cohort_min_items"]),
        item_codes=codes, execution_order=[cid for cid, _ in combined], workers=assign,
        cohort_of={cid: co for cid, co in combined},                     # kept ONLY in this file; reviewers never receive it
        reviewer={cid: T["assignment"]["reviewer_for"][assign[cid]] for cid in assign},
        successor={cid: T["assignment"]["successor_for"][assign[cid]] for cid in assign})
    json.dump(sel, open(STUDY / "selection.json", "w"), indent=1)
    print(f"F-only {len(f_only)}  P-only {len(p_only)}  cohort_ok={sel['cohort_ok']}  shared {len(shared)}")
    for cid in p_only:
        print("  P-only", codes[cid], cid, f"formula_rank={frank[cid]}", cards[cid]["title"][:60])
    for cid in f_only:
        print("  F-only", codes[cid], cid, f"formula_rank={frank[cid]}", cards[cid]["title"][:60])


if __name__ == "__main__":
    main()

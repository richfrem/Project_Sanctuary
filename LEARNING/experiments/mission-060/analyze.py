#!/usr/bin/env python3
"""Analysis and verdict for the MISSION-060 study. Applies the frozen decision table in protocol.md; nothing here may be tuned after freeze.

Inputs : thresholds.json, selection.json, assess/<ITEM-NN>.json, usage-log.jsonl, calibration-results.json, declarations/attestation.json,
         scenario-results-heldout.json.
Output : results.json (per-claim results, cohort tables, confounds, verdict) and a printed summary.
"""
import glob
import json
import pathlib
import re

from common import usage_totals, item_usage

STUDY = pathlib.Path(__file__).resolve().parent
T = json.load(open(STUDY / "thresholds.json"))


def load(name, default=None):
    p = STUDY / name
    return json.loads(p.read_text()) if p.exists() else default


def is_spo(a):
    """Substantive preserved output: every check of its type must hold (partial outputs are reported, not counted)."""
    if not a["deterministic"].get("files"):
        return False, dict(files=False)
    checks = (a.get("review") or {}).get("checks")
    required = {f"S{i}" for i in range(1, 6)} | (set(("C2a", "C2b", "C2c", "C2d", "C3"))
                                                    if a["kind"] == "creative" else {"D1", "D2", "D3"})
    if not isinstance(checks, dict) or not required.issubset(checks):
        raise ValueError("review is missing required checks")
    ck = {}
    for key in required:
        entry = checks[key]
        if not isinstance(entry, dict) or type(entry.get("value")) is not bool:
            raise ValueError("review values must be JSON booleans")
        ck[key] = entry["value"]
    s_ok = sum(1 for i in range(1, 6) if ck.get(f"S{i}")) >= 4
    det = a["deterministic"]
    if a["kind"] == "creative":
        parts = dict(C1=det.get("brief_predeclared"), C2=all(ck.get(k) for k in ("C2a", "C2b", "C2c", "C2d")), C3=ck.get("C3") and det.get("reflection_present"),
                     files=det.get("findings_present"), S=s_ok)
    else:
        parts = dict(D1=ck.get("D1"), D2=ck.get("D2"), D3=ck.get("D3"), files=det.get("findings_present"), S=s_ok)
    return all(bool(v) for v in parts.values()), parts


def main():
    sel = load("selection.json")
    cal = load("calibration-results.json", {})
    att = load("declarations/attestation.json", {})
    scen = load("scenario-results-heldout.json", {})
    log_path = STUDY / "usage-log.jsonl"
    log = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()] if log_path.exists() else []
    total, unknown_calls = usage_totals(log)

    items = {}
    for p in sorted(glob.glob(str(STUDY / "assess" / "ITEM-*.json"))):
        a = json.loads(pathlib.Path(p).read_text())
        try:
            spo, parts = is_spo(a)
            review_valid, review_error = True, None
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            spo, parts = False, dict(invalid_review=False)
            review_valid, review_error = False, str(error)
        expected = None if a.get("deterministic", {}).get("files") else {f"work-{a['code']}"}
        w = item_usage(log, a["code"], expected=expected)
        items[a["code"]] = dict(id=a["id"], kind=a["kind"], worker=a["worker"], cohort=sel["cohort_of"][a["id"]], spo=spo, parts=parts,
                                additional_tokens=w.get("additional_tokens"), cap_exceeded=(w.get("additional_tokens") or 0) > T["per_item_cap_additional_tokens"],
                                usage_complete=w["calls_complete"] and not w["unknown_calls"],
                                review_valid=review_valid, review_error=review_error,
                                status=a.get("worker_status"))

    def cohort(c):
        rows = [v for v in items.values() if v["cohort"] == c]
        toks = [v["additional_tokens"] for v in rows]
        known = all(type(t) is int and t > 0 for t in toks) and all(row["usage_complete"] for row in rows) and bool(rows)
        return dict(items=len(rows), spo=sum(v["spo"] for v in rows), tokens=(sum(toks) if known else None), tokens_known=known,
                    spo_by_kind={k: sum(1 for v in rows if v["kind"] == k and v["spo"]) for k in ("discovery", "creative")},
                    spo_by_worker={w: sum(1 for v in rows if v["worker"] == w and v["spo"]) for w in T["assignment"]["workers"]},
                    cap_exceeded=sum(v["cap_exceeded"] for v in rows))
    P, F = cohort("P-only"), cohort("F-only")

    # validity gates
    gates = dict(
        calibration_reliability=(cal.get("agreement") is not None and cal["agreement"] >= T["calibration_agreement_min"]),
        declarations_blind=att.get("blind") is True,
        cohort_sizes=(P["items"] >= T["cohort_min_items"] and F["items"] >= T["cohort_min_items"]),
        usage_measured=(P["tokens_known"] and F["tokens_known"] and bool(log) and not unknown_calls),
        review_contract=bool(items) and all(row["review_valid"] for row in items.values()),
        item_caps=all(not row["cap_exceeded"] for row in items.values()),
        execution_limit=len(items) <= T["max_executed_items"],
        attempt_records_complete=all(match.group(1) in items for record in log
                                     if (match := re.fullmatch(r"work-(ITEM-\d{2})", record["call_id"]))))

    # claim B
    B = {}
    if gates["usage_measured"] and gates["cohort_sizes"]:
        rp, rf = P["spo"] / P["tokens"], F["spo"] / F["tokens"]
        ratio_ok = (P["spo"] > 0) if F["spo"] == 0 else (rp / rf >= T["B1_min_ratio_spo_per_budget"])
        B["B1"] = dict(met=bool(ratio_ok and (P["spo"] - F["spo"] >= T["B1_min_absolute_spo_difference"])), spo_P=P["spo"], spo_F=F["spo"],
                       per_100k_P=round(rp * 1e5, 3), per_100k_F=round(rf * 1e5, 3), ratio=(None if F["spo"] == 0 else round(rp / rf, 3)),
                       ratio_ok=bool(ratio_ok), abs_diff_ok=(P["spo"] - F["spo"] >= T["B1_min_absolute_spo_difference"]))
    else:
        B["B1"] = dict(met=None, reason="validity gate not satisfied")
    B["B2"] = dict(met=(None if unknown_calls else total <= T["total_additional_tokens_ceiling"]),
                   total_additional_tokens=(None if unknown_calls else total), known_subtotal=total, ceiling=T["total_additional_tokens_ceiling"],
                   calls=len(log), calls_without_usage=unknown_calls)
    B["B3"] = dict(met=True, note="Ranks 1-11 are the same items under F and P and are not executed; the formula share is retained by construction, not measured. "
                                  "The real diversion cost is the F-only cohort, reported here.", diversion_cost_spo=F["spo"])

    A = scen.get("claim_A", "not run")
    if not isinstance(A, str):
        A = "unrecognized claim-A evidence"
    if not all(gates.values()):
        verdict = "UNASSESSED"
        why = [k for k, v in gates.items() if not v]
    elif B["B1"]["met"] is False or B["B2"]["met"] is False or A.startswith("FALSIFIED"):
        verdict, why = "FALSIFIED", [k for k, v in (("B1", B["B1"]["met"]), ("B2", B["B2"]["met"])) if v is False] + (["A"] if A.startswith("FALSIFIED") else [])
    elif A.startswith("UNDECIDED") or A == "not run":
        verdict, why = "UNASSESSED", ["claim A undecided or not run"]
    elif B["B1"]["met"] is True and B["B2"]["met"] is True and (A == "MET" or A.startswith("UNSCORABLE")):
        verdict, why = "CORROBORATED", []
    else:
        verdict, why = "UNASSESSED", ["claim results missing or unrecognized"]
    res = dict(verdict=verdict, why=why, validity_gates=gates, claim_B=B, claim_A=dict(result=A, summary=scen.get("summary"),
               bootstrap_pass_probability=scen.get("bootstrap_pass_probability")), cohorts=dict(P_only=P, F_only=F), items=items,
               natural_observation=dict(note="reported separately; does not affect the verdict", target=T["natural_observation"]["target_count"]))
    (STUDY / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("verdict", "why", "validity_gates")}, indent=1))
    print("B1:", json.dumps(B["B1"]))
    print("B2:", B["B2"]["met"], total, "calls", len(log), "unknown", len(unknown_calls))
    print("claim A:", A)
    print("cohorts:", json.dumps({"P_only": {k: P[k] for k in ("items", "spo", "tokens")}, "F_only": {k: F[k] for k in ("items", "spo", "tokens")}}))


if __name__ == "__main__":
    main()

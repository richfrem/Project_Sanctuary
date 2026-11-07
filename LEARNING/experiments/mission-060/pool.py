#!/usr/bin/env python3
"""Deterministic candidate pool and novelty check for the MISSION-060 study.

make_base(): the stripped candidate cards for existing missions and inbox ideas (NO scores, NO ranks) and, separately, the
formula inputs of the missions. Declarers only ever see the cards.
containment(): fraction of a text's word 3-grams already present in one knowledge-base document (max over documents).
"""
import glob
import json
import pathlib
import re

STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[2]
T = json.load(open(STUDY / "thresholds.json"))


def sentences(text, n=2, limit=420):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return " ".join(parts[:n])[:limit]


def make_base():
    cards, formula = [], {}
    for line in open(REPO / "LEARNING/index.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["status"] != "PROPOSED" or r["id"] in T["excluded_from_pool"]:
            continue
        m = json.load(open(REPO / f"LEARNING/missions/{r['id']}.json", encoding="utf-8"))
        if m.get("human_gate_required"):
            continue
        cards.append(dict(id=m["id"], kind="discovery", source="mission", title=m["title"], summary=str(m["hypothesis"])[:600],
                          first_unit=sentences(m.get("experiment", "Inspect the mission's stated prerequisites and report what is ready."))))
        s = m["scores"]
        formula[m["id"]] = dict(strategic_leverage=s["strategic_leverage"], epistemic_depth=s["epistemic_depth"],
                                cost_weight=s["cost_weight"], operational_friction=s["operational_friction"], source="manifest")
    for line in open(REPO / "LEARNING/inbox.jsonl", encoding="utf-8"):
        o = json.loads(line)
        cards.append(dict(id=o["id"], kind="discovery", source="inbox", title=o["idea"][:80], summary=o["idea"][:600],
                          first_unit="Scope the idea and take the smallest step that shows whether it is worth doing; report what you find."))
    cards.sort(key=lambda c: c["id"])
    return cards, formula


def _words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3]


def _shingles(text):
    w = _words(text)
    return {" ".join(w[i:i + 3]) for i in range(len(w) - 2)}


def kb_docs(exclude_ids=()):
    docs = {}
    for p in glob.glob(str(REPO / "LEARNING/missions/MISSION-*.json")):
        m = json.load(open(p, encoding="utf-8"))
        if m["id"] in exclude_ids:
            continue
        docs[m["id"]] = _shingles(m["title"] + " " + str(m.get("hypothesis", "")))
    for p in glob.glob(str(REPO / "LEARNING/concepts/*.md")):
        docs[p] = _shingles(open(p, encoding="utf-8").read()[:6000])
    for p in glob.glob(str(REPO / "chronicle/entries/*.md")):
        docs[p] = _shingles(open(p, encoding="utf-8").read()[:3000])
    return docs


def containment(card, docs):
    s = _shingles(card["title"] + " " + card["summary"])
    if not s:
        return 0.0
    return max((len(s & d) / len(s) for d in docs.values()), default=0.0)


if __name__ == "__main__":
    cards, formula = make_base()
    json.dump(cards, open(STUDY / "pool-base.json", "w"), indent=1)
    json.dump(formula, open(STUDY / "formula-base.json", "w"), indent=1)
    print(f"pool-base.json: {len(cards)} cards ({sum(c['source']=='mission' for c in cards)} missions, {sum(c['source']=='inbox' for c in cards)} inbox); formula-base.json: {len(formula)}")

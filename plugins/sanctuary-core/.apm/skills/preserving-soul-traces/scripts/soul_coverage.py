#!/usr/bin/env python3
"""soul_coverage.py - coverage-aware soul export and verification (MISSION-055 follow-up: soul preservation repair).

The older closure command (`trace_extractor.py --source-dir chronicle/entries --append`) collected Chronicle Markdown only; mission manifests,
experiment findings, the event ledger (handoffs, verdicts, failures, repairs, lessons, interventions) and supporting artifacts never reached the
dataset. This tool applies an explicit, declarative policy (`references/coverage-policy.json`) to the TRACKED files of the repository, writes
records in the ADR 084 canonical schema, and writes a source inventory (`huggingface/metadata/coverage_manifest.json`) so completeness is checkable.

    soul_coverage.py export [--dry-run]   apply the policy: add/update records (idempotent), write the coverage manifest
    soul_coverage.py check                report four separate facts: SCHEMA, COVERAGE, LOCAL EXPORT, REMOTE; exit code says which failed
    soul_coverage.py inventory            print the per-category table (discovered / included / excluded / failed / uncategorized)
    soul_coverage.py closure              THE closure command: export + the four-fact report (use after a mission is merged)
    soul_coverage.py status               one-line status (used by guardian_wakeup)

Exit codes of `check`: 0 all four verified; 2 coverage incomplete; 3 local export stale or schema invalid; 4 local export ready but remote persistence not verified.
Stdlib only. Knowledge goes into records; supporting artifacts are stable references (path, bytes, sha256, git commit), never ephemeral paths.
"""
import argparse, fnmatch, hashlib, json, re, subprocess, sys, time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trace_extractor as TE  # noqa: E402

EXPORT_SCHEMA = "soul-coverage/1"
EXTRACTOR_VERSION = "2"
DATASET = Path("huggingface/data/soul_traces.jsonl")
MANIFEST = Path("huggingface/metadata/coverage_manifest.json")
RECEIPTS = Path(".agent/learning/soul_sync_receipts.jsonl")
POLICY = HERE.parent / "references" / "coverage-policy.json"
SECRET_RX = [re.compile(p) for p in (r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{20,}", r"Bearer\s+[A-Za-z0-9._\-]{16,}", r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{6,}",
                                       r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|secret)\"?\s*[:=]\s*\"?[A-Za-z0-9._\-]{20,}", r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}", r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}",
                                       r"github_pat_[A-Za-z0-9_]{22,}", r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])", r"xox[abprs]-[A-Za-z0-9-]{10,}",
                                       r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----", r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_\-]{35}")]
# Local account paths are personal data in a PUBLIC dataset: replaced, not refused, so provenance text survives.
LOCAL_PATH_RX = [(re.compile(r"/Users/[^/\s\"'`<>:)\]]+"), "/Users/<user>"), (re.compile(r"/home/[^/\s\"'`<>:)\]]+"), "/home/<user>")]


def scrub_local_paths(text):
    for rx, repl in LOCAL_PATH_RX:
        text = rx.sub(repl, text)
    return text
EVENT_CHUNK = 40


def sh(root, *a):
    return subprocess.run(a, cwd=root, capture_output=True, text=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def glob_rx(pat):
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out += "(?:.*/)?"; i += 3
        elif pat.startswith("**", i):
            out += ".*"; i += 2
        elif pat[i] == "*":
            out += "[^/]*"; i += 1
        elif pat[i] == "?":
            out += "[^/]"; i += 1
        else:
            out += re.escape(pat[i]); i += 1
    return re.compile("^" + out + "$")


def load_policy():
    p = json.loads(POLICY.read_text())
    for r in p["rules"]:
        r["_rx"] = [glob_rx(m) for m in r["match"]]
    return p


def classify(path, policy):
    for r in policy["rules"]:
        if any(rx.match(path) for rx in r["_rx"]):
            return r
    return None


def tracked_files(root):
    return [l for l in sh(root, "git", "ls-files").splitlines() if l]


def last_commits(root):
    """path -> most recent commit touching it (one pass over history)."""
    m, cur = {}, None
    for line in sh(root, "git", "log", "--pretty=format:@@%H", "--name-only").splitlines():
        if line.startswith("@@"):
            cur = line[2:]
        elif line and line not in m:
            m[line] = cur
    return m


def reachable_from_remote(root, commit, cache):
    """True only if the commit is an ancestor of origin/main (a stable, published reference); a local-only commit is NOT a stable reference."""
    if commit not in cache:
        cache[commit] = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=root, capture_output=True).returncode == 0
    return cache[commit]


def has_secret(text):
    return next((rx.pattern[:24] for rx in SECRET_RX if rx.search(text)), None)


def make_record(rec_id, content, source, snapshot_type, model_version, ts=None):
    content = scrub_local_paths(content)
    return {"id": rec_id, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(), "timestamp": ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_version": model_version, "snapshot_type": snapshot_type, "valence": 0.5, "uncertainty": 0.1, "semantic_entropy": 0.5, "alignment_score": 0.85,
            "stability_class": "STABLE", "adr_version": "084", "content": content, "source_file": source}


# ---------------------------------------------------------------- handlers: return list of (record_id, content) for one source
def h_mission(path, text, src_sha):
    m = json.loads(text)
    L = [f"# {m['id']}: {m.get('title', '')}", "",
         f"_Representation: deterministic extraction of `{path}` (source sha256 {src_sha[:16]}); fields are copied, not interpreted._", "",
         f"- **Status:** {m.get('status')}  **Verdict:** {m.get('verdict')}  **Surface:** {m.get('execution_surface')}  **Priority:** {m.get('priority_score')}",
         f"- **Created/updated:** {m.get('created_at')} / {m.get('updated_at')}", "", "## Hypothesis", m.get("hypothesis", ""), "", "## Falsification criteria", str(m.get("falsification_criteria", "")), ""]
    for k, t in (("experiment", "Experiment / method"), ("baseline_evidence", "Baseline evidence"), ("target_platform", "Target")):
        if m.get(k):
            L += [f"## {t}", str(m[k]), ""]
    for k, t in (("evidence", "Evidence"), ("lessons_learned", "Lessons learned"), ("design_constraints", "Design constraints / limitations")):
        v = m.get(k)
        if v:
            L += [f"## {t}"] + [f"- {json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x}" for x in v] + [""]
    if m.get("related_missions"):
        L += ["## Related missions"] + [f"- {k}: {v}" for k, v in m["related_missions"].items()] + [""]
    L += ["## Other fields (verbatim JSON)", "```json", json.dumps({k: v for k, v in m.items() if k not in ("id", "title", "status", "verdict", "execution_surface", "priority_score", "created_at", "updated_at", "hypothesis", "falsification_criteria", "experiment", "baseline_evidence", "target_platform", "evidence", "lessons_learned", "design_constraints", "related_missions")}, ensure_ascii=False, indent=1, sort_keys=True), "```"]
    return [(f"mission__{m['id']}", "\n".join(L))]


def h_events(path, text, src_sha):
    groups = OrderedDict()
    bad, skipped = 0, []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError:
            bad += 1; continue
        if has_secret(line):
            skipped.append(e.get("timestamp") or "?"); continue
        groups.setdefault(str(e.get("event_type") or e.get("kind") or "OTHER"), []).append(e)
    out = []
    for typ, evs in groups.items():
        for i in range(0, len(evs), EVENT_CHUNK):
            part = evs[i:i + EVENT_CHUNK]
            head = [f"# Event ledger digest: {typ} (events {i + 1}-{i + len(part)} of {len(evs)})", "",
                    f"_Representation: verbatim JSON events from `{path}` in file order (no summarisation). Types preserved: handoffs (CHECKIN), verdicts (MISSION_RESOLVED), failures and repairs, interventions, lessons, deliveries, evaluations, costs, procedure outcomes._", ""]
            out.append((f"events__{typ}__{i // EVENT_CHUNK + 1:03d}", "\n".join(head) + "\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in part)))
    return out, bad, skipped


def h_artifacts(group, files, root, commits, reach):
    rows = []
    for p in sorted(files):
        b = (root / p).read_bytes()
        c = commits.get(p)
        r = reachable_from_remote(root, c, reach) if c else False
        rows.append({"path": p, "bytes": len(b), "sha256": sha(b), "git_commit": c, "commit_on_origin_main": r})
    L = [f"# Supporting artifacts: {group}", "", "_Representation: reference index (not file contents). Each artifact is recoverable with `git show <git_commit>:<path>` from the public repository `richfrem/Project_Sanctuary`; the sha256 verifies the bytes. `commit_on_origin_main` must be true for a stable reference; false means the commit has not been published yet._", "",
         "| path | bytes | sha256 | git_commit | on origin/main |", "|---|---|---|---|---|"]
    L += [f"| `{x['path']}` | {x['bytes']} | {x['sha256']} | {x['git_commit']} | {x['commit_on_origin_main']} |" for x in rows]
    return f"artifacts__{re.sub(r'[^A-Za-z0-9]+', '_', group).strip('_')}", "\n".join(L), rows


def load_dataset(root):
    recs = OrderedDict()
    p = root / DATASET
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); recs[r["id"]] = r
    return recs


def write_dataset(root, recs):
    p = root / DATASET; p.parent.mkdir(parents=True, exist_ok=True)
    # Never expose a truncated JSONL if interrupted during automatic capture.
    import os, tempfile
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".soul-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs.values()))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def export(root, dry=False, model_version="Sanctuary-Qwen2-7B-v1.0-GGUF-Final"):
    pol = load_policy()
    files = tracked_files(root)
    recs = load_dataset(root)
    by_sha = {r["sha256"]: r["id"] for r in recs.values()}
    commits, reach = last_commits(root), {}
    entries, uncategorized, counts = [], [], Counter()
    watched = tuple(pol["watched_roots"])
    artifact_groups = OrderedDict()
    for path in files:
        rule = classify(path, pol)
        if rule is None:
            if path.startswith(watched):
                uncategorized.append(path)
            continue
        e = {"path": path, "category": rule["name"], "required": bool(rule.get("required"))}
        full = root / path
        try:
            raw = full.read_bytes()
        except OSError as ex:
            e.update(status="failed", reason=f"unreadable: {ex}"); entries.append(e); continue
        e.update(bytes=len(raw), source_sha256=sha(raw))
        if rule.get("action") == "exclude":
            e.update(status="excluded", reason=rule["reason"]); entries.append(e); continue
        h = rule["handler"]
        if h == "artifact_refs":
            depth = rule.get("group_depth", 3)
            artifact_groups.setdefault((rule["name"], "/".join(path.split("/")[:depth])), []).append((path, e)); entries.append(e); continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            e.update(status="failed", reason="not valid UTF-8 text (unsupported type for a required text source)"); entries.append(e); continue
        hit = None if h == "events_digest" else has_secret(text)          # the event ledger is scanned line by line instead
        if hit:
            e.update(status="excluded_sensitive", reason=f"secret-like pattern ({hit}...) found; refusing to export"); entries.append(e); continue
        if len(raw) > pol["max_record_bytes"]:
            e.update(status="failed", reason=f"exceeds max_record_bytes ({pol['max_record_bytes']}); needs chunking or an artifact reference"); entries.append(e); continue
        try:
            if h == "markdown":
                rec = TE.build_soul_trace_record(full, root, model_version, rule["snapshot_type"])
                if rec is None:
                    e.update(status="excluded", reason="empty after sanitisation"); entries.append(e); continue
                items = [(rec["id"], rec["content"], rec)]
                if rule.get("dedupe_by_content") and rec["sha256"] in by_sha and by_sha[rec["sha256"]] != rec["id"]:
                    e.update(status="included", action={"duplicate_content": 1}, record_ids=[by_sha[rec["sha256"]]], reason="identical content already preserved under another source path"); entries.append(e); continue
            elif h == "mission_json":
                items = [(i, c, None) for i, c in h_mission(path, text, e["source_sha256"])]
            elif h == "events_digest":
                lst, bad, skipped = h_events(path, text, e["source_sha256"]); items = [(i, c, None) for i, c in lst]
                if bad:
                    e["malformed_lines_skipped"] = bad
                if skipped:
                    e["sensitive_events_skipped"] = len(skipped)
            else:
                raise ValueError(f"unknown handler {h}")
        except Exception as ex:  # noqa: BLE001
            e.update(status="failed", reason=f"parse/handler error: {type(ex).__name__}: {ex}"); entries.append(e); continue
        ids, actions = [], Counter()
        for rid, content, prebuilt in items:
            new = prebuilt or make_record(rid, content, path, rule["snapshot_type"], model_version)
            if scrub_local_paths(new["content"]) != new["content"]:       # prebuilt (markdown) records bypass make_record
                new = {**new, "content": scrub_local_paths(new["content"])}
                new["sha256"] = hashlib.sha256(new["content"].encode("utf-8")).hexdigest()
            old = recs.get(rid)
            if old and old["sha256"] == new["sha256"]:
                actions["unchanged"] += 1
            else:
                actions["updated" if old else "added"] += 1
                if not dry:
                    recs[rid] = new
            ids.append(rid)
        e.update(status="included", action=dict(actions), record_ids=ids, record_sha256=[(recs.get(i) or {}).get("sha256") for i in ids]); entries.append(e)
    for (cat, group), members in artifact_groups.items():
        rid, content, rows = h_artifacts(group, [p for p, _ in members], root, commits, reach)
        old = recs.get(rid); new = make_record(rid, content, group, "learning", model_version)
        act = "unchanged" if old and old["sha256"] == new["sha256"] else ("updated" if old else "added")
        if act != "unchanged" and not dry:
            recs[rid] = new
        for (p, e), row in zip(sorted(members, key=lambda t: t[0]), rows):
            e.update(status="included", action={act: 1}, record_ids=[rid], artifact_ref={"repo": "richfrem/Project_Sanctuary", "git_commit": row["git_commit"], "commit_on_origin_main": row["commit_on_origin_main"], "sha256": row["sha256"]})
    for e in entries:
        counts[(e["category"], e["status"])] += 1
    idx_id, idx_content = h_index(entries, recs, uncategorized, root)
    old_idx = recs.get(idx_id); new_idx = make_record(idx_id, idx_content, "huggingface/metadata/coverage_manifest.json", "learning", model_version)
    if not old_idx or old_idx["sha256"] != new_idx["sha256"]:
        if not dry:
            recs[idx_id] = new_idx
    manifest = {"export_schema": EXPORT_SCHEMA, "index_record_id": idx_id, "extractor_version": EXTRACTOR_VERSION, "policy_version": pol["policy_version"], "policy_sha256": sha(POLICY.read_bytes()),
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "base_commit": sh(root, "git", "rev-parse", "HEAD").strip(),
                "dataset_records": len(recs), "uncategorized_under_watched_roots": uncategorized, "sources": sorted(entries, key=lambda x: x["path"])}
    manifest["summary"] = summarize(manifest, pol)
    if not dry:
        write_dataset(root, recs)
        TE.update_manifest(root / "huggingface/metadata/manifest.json", root / DATASET, model_version)
        (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
        from preservation_workflow import atomic_json
        atomic_json(root / MANIFEST, manifest, indent=1)
    return manifest


def h_index(entries, recs, uncategorized, root):
    """Recovery aid for a successor: ONE record that lists every mission with its verdict, the records that hold its knowledge, and where its
    supporting artifacts live. Deterministic (no timestamp); rebuilt from the same inventory as the manifest."""
    by_mission, missions = {}, {}
    for e in entries:
        p = e["path"]
        if e["category"] == "mission_manifests" and e.get("record_ids"):
            mid = e["record_ids"][0].split("__", 1)[1]; missions.setdefault(mid, {})["manifest_path"] = p; missions[mid]["record"] = e["record_ids"][0]
        m = re.match(r"LEARNING/experiments/mission-(\d+)", p)
        if m and e.get("record_ids"):
            by_mission.setdefault(f"MISSION-{int(m.group(1)):03d}", set()).update(e["record_ids"])
    verdicts = {}
    for mid, v in missions.items():
        try:
            mj = json.loads((root / v["manifest_path"]).read_text()) if (root / v["manifest_path"]).exists() else {}
        except Exception:  # noqa: BLE001
            mj = {}
        verdicts[mid] = (mj.get("status"), mj.get("verdict"), mj.get("title", ""))
    L = ["# Soul index: how to find and recover preserved learning", "",
         "_Deterministic index built from `huggingface/metadata/coverage_manifest.json`. Knowledge lives in the records named below; supporting artifacts (code, raw results) are referenced in `artifacts__*` records by path, sha256 and git commit of the public repository `richfrem/Project_Sanctuary` (recover with `git show <commit>:<path>`). Event-ledger digests (`events__<TYPE>__NNN`) hold handoffs, verdicts, failures, repairs, interventions and lessons verbatim._", "",
         "| mission | status | verdict | mission record | experiment/finding/artifact records |", "|---|---|---|---|---|"]
    for mid in sorted(missions):
        st, vd, title = verdicts.get(mid, (None, None, ""))
        extra = ", ".join(sorted(by_mission.get(mid, []))) or "-"
        L.append(f"| {mid} {title[:70].replace('|', '/')} | {st} | {vd} | {missions[mid]['record']} | {extra} |")
    L += ["", f"Uncategorized files under watched roots (not preserved, listed for review): {len(uncategorized)}"]
    return "soul_index", "\n".join(L)


def summarize(manifest, pol):
    cats = OrderedDict()
    for r in pol["rules"]:
        cats[r["name"]] = {"required": bool(r.get("required")), "discovered": 0, "included": 0, "excluded": 0, "failed": 0, "excluded_sensitive": 0}
    for e in manifest["sources"]:
        c = cats[e["category"]]; c["discovered"] += 1
        c[{"included": "included", "excluded": "excluded", "failed": "failed", "excluded_sensitive": "excluded_sensitive"}[e["status"]]] += 1
    problems = []
    for e in manifest["sources"]:
        if e.get("malformed_lines_skipped") or e.get("sensitive_events_skipped"):
            problems.append(f"{e['path']}: {e.get('malformed_lines_skipped', 0)} malformed and {e.get('sensitive_events_skipped', 0)} sensitive lines were not exported")
        if e["required"] and e["status"] in ("failed", "excluded_sensitive"):
            problems.append(f"required source {e['status']}: {e['path']} ({e['reason']})")
        if e["required"] and e["status"] == "excluded" and "empty" not in e["reason"]:
            problems.append(f"required source excluded: {e['path']} ({e['reason']})")
        ar = e.get("artifact_ref")
        if ar and not ar["commit_on_origin_main"]:
            problems.append(f"artifact reference not yet on origin/main (publish first): {e['path']}")
    # a required rule with zero discovered sources would mean the policy no longer matches the repository
    for n, c in cats.items():
        if c["required"] and c["discovered"] == 0:
            problems.append(f"required category '{n}' matched no tracked file")
    return {"by_category": cats, "problems": problems, "required_ok": not problems, "uncategorized_count": len(manifest["uncategorized_under_watched_roots"])}


def last_receipt(root):
    p = root / RECEIPTS
    if not p.exists():
        return None
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def check(root):
    """Four separate facts. Never prints an unqualified 'persisted' unless all four hold."""
    out = {"schema": None, "coverage": None, "local_export": None, "remote": "NOT VERIFIED", "problems": []}
    ok, errs = TE.verify_dataset(root / DATASET)
    out["schema"] = "valid" if ok else "INVALID"
    if not ok:
        out["problems"] += [f"schema: {x}" for x in errs[:5]]
    if not (root / MANIFEST).exists():
        out["coverage"] = "NO COVERAGE MANIFEST"; out["local_export"] = "missing"; out["problems"].append("coverage manifest missing: run soul_coverage.py export")
        return out
    man = json.loads((root / MANIFEST).read_text())
    now = export(root, dry=True)          # recompute from current tracked sources without writing
    stale = []
    old = {e["path"]: e for e in man["sources"]}
    new = {e["path"]: e for e in now["sources"]}
    for p, e in new.items():
        o = old.get(p)
        if e.get("status") == "excluded":                    # policy-EXCLUDED files (receipts, code/tests versioned in git, derived state) never make the export stale
            continue
        if o is None:
            stale.append(f"new source not in manifest: {p}")
        elif o.get("source_sha256") != e.get("source_sha256"):
            stale.append(f"source changed since export: {p}")
    for p in old:
        if p not in new and old[p].get("status") != "excluded":
            stale.append(f"source gone from repository: {p}")
    recs = load_dataset(root)
    for e in man["sources"]:
        for rid, rs in zip(e.get("record_ids", []), e.get("record_sha256", [None] * len(e.get("record_ids", [])))):
            if rid not in recs:
                stale.append(f"record {rid} for {e['path']} missing from dataset")
            elif rs and recs[rid]["sha256"] != rs:
                stale.append(f"INTEGRITY: record {rid} differs from what the manifest recorded for {e['path']}")
    cov = man["summary"]
    out["coverage"] = "complete" if cov["required_ok"] and not now["summary"]["problems"] else "INCOMPLETE"
    out["problems"] += (cov["problems"] or []) + [p for p in now["summary"]["problems"] if p not in cov["problems"]]
    out["uncategorized"] = cov["uncategorized_count"]
    out["local_export"] = "current" if not stale else "STALE"
    out["problems"] += stale[:20]
    ds = hashlib.sha256((root / DATASET).read_bytes()).hexdigest()
    rc = last_receipt(root)
    if rc and rc.get("verified") and rc.get("local_dataset_sha256") == ds and rc.get("local_manifest_sha256") == sha((root / MANIFEST).read_bytes()):
        out["remote"] = f"verified at revision {rc['revision'][:10]} ({rc['time']})"
    else:
        out["remote"] = "NOT VERIFIED" + (" (last receipt is for a different dataset version)" if rc else " (no verified sync receipt)")
    return out


def sources_changed_since_export(root):
    """Read-only: preserved sources committed since the manifest's base commit; None when there is no manifest."""
    root = Path(root)
    mp = root / MANIFEST
    if not mp.exists():
        return None
    base = json.loads(mp.read_text()).get("base_commit", "")
    changed = [p for p in sh(root, "git", "diff", "--name-only", base, "HEAD").splitlines() if p] if base else []
    pol = load_policy()
    return [p for p in changed if (r := classify(p, pol)) is not None and r.get("action") != "exclude"]


def quick_status(root):
    """Cheap one-line status for session boot (no full recomputation): manifest age in commits, sources changed since export, last remote receipt."""
    root = Path(root)
    mp = root / MANIFEST
    if not mp.exists():
        return "WARN", "soul coverage manifest missing: run soul_coverage.py closure"
    man = json.loads(mp.read_text())
    base = man.get("base_commit", "")
    rel = sources_changed_since_export(root)
    ds = hashlib.sha256((root / DATASET).read_bytes()).hexdigest() if (root / DATASET).exists() else ""
    rc = last_receipt(root)
    remote = "remote verified" if rc and rc.get("verified") and rc.get("local_dataset_sha256") == ds and rc.get("local_manifest_sha256") == sha(mp.read_bytes()) else "remote NOT verified"
    cov = "coverage complete" if man["summary"]["required_ok"] else "coverage INCOMPLETE"
    if rel:
        return "WARN", f"soul export STALE: {len(rel)} preserved source(s) changed since export {base[:8]} (run soul_coverage.py closure); {cov}; {remote}"
    return ("PASS" if remote == "remote verified" and cov == "coverage complete" else "WARN"), f"soul export current at {base[:8]}; {cov}; {remote}"


def report(res):
    print(f"SCHEMA        : {res['schema']}")
    print(f"COVERAGE      : {res['coverage']}  (uncategorized files under watched roots: {res.get('uncategorized', '?')})")
    print(f"LOCAL EXPORT  : {res['local_export']}")
    print(f"REMOTE        : {res['remote']}")
    for p in res["problems"][:25]:
        print(f"  - {p}")
    if len(res["problems"]) > 25:
        print(f"  ... and {len(res['problems']) - 25} more")
    good = res["schema"] == "valid" and res["coverage"] == "complete" and res["local_export"] == "current"
    if good and res["remote"].startswith("verified"):
        print("SOUL PERSISTED (schema valid, required coverage complete, local export current, remote verified).")
        return 0
    if res["coverage"] != "complete":
        print("SOUL PARTIAL: required coverage is incomplete; do not report the soul as preserved.")
        return 2
    if res["schema"] != "valid" or res["local_export"] != "current":
        print("SOUL EXPORT NOT CURRENT: re-run soul_coverage.py export (and commit the sources) before syncing.")
        return 3
    print("SOUL EXPORTED LOCALLY: coverage complete and current, but REMOTE PERSISTENCE IS NOT VERIFIED (not yet synced, or not yet confirmed).")
    return 4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "check", "inventory", "closure", "status"]); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--repo-root", default=".")
    a = ap.parse_args(argv)
    root = Path(a.repo_root).resolve()
    if a.cmd == "export":
        from preservation_workflow import preservation_lock
        with preservation_lock(root):
            m = export(root, dry=a.dry_run)
        s = m["summary"]
        acts = Counter()
        for e in m["sources"]:
            for k, v in (e.get("action") or {}).items():
                acts[k] += v
        print(f"export {'(dry run) ' if a.dry_run else ''}: {len(m['sources'])} policy-matched sources, {m['dataset_records']} dataset records; actions {dict(acts)}; uncategorized {s['uncategorized_count']}")
        print("required coverage:", "complete" if s["required_ok"] else "INCOMPLETE")
        for p in s["problems"][:20]:
            print("  -", p)
        return 0 if s["required_ok"] else 2
    if a.cmd == "status":
        st, msg = quick_status(root); print(f"{st}: {msg}"); return 0 if st == "PASS" else 1
    if a.cmd == "closure":            # the documented session/mission closure command: export, then report the four facts
        from preservation_workflow import run
        state = run(root)
        print("LOCAL PRESERVATION:", state["status"])
        if state["status"] != "local_complete":
            for p in state.get("problems", []):
                print("  -", p)
            print("Resume:", state["resume"])
            return 2
        return report(state["facts"])
    if a.cmd == "inventory":
        m = json.loads((root / MANIFEST).read_text()) if (root / MANIFEST).exists() else export(root, dry=True)
        print(f"{'category':38s} {'req':3s} {'found':>6s} {'incl':>6s} {'excl':>6s} {'fail':>5s}")
        for n, c in m["summary"]["by_category"].items():
            print(f"{n:38s} {'Y' if c['required'] else '-':3s} {c['discovered']:6d} {c['included']:6d} {c['excluded'] + c['excluded_sensitive']:6d} {c['failed']:5d}")
        return 0
    return report(check(root))


if __name__ == "__main__":
    sys.exit(main())

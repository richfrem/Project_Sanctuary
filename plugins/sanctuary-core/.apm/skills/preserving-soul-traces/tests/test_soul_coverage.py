"""Coverage-aware soul export: fixtures in a throwaway git repository with a local 'origin' (no network, no Hugging Face)."""
import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import soul_coverage as S
import sync_soul_dataset as SYNC
import trace_extractor as TE


def git(root, *a):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=root, capture_output=True, text=True, check=True).stdout


EVENTS = [
    {"event_type": "CHECKIN", "mission_id": "MISSION-901", "agent_id": "a", "handoff": "HANDOFF-TEXT-901 do not redo X; next: Y", "timestamp": "2026-01-01T00:00:00Z"},
    {"event_type": "MISSION_RESOLVED", "mission_id": "MISSION-901", "verdict": "FALSIFIED", "lessons": ["LESSON-TEXT-901"], "timestamp": "2026-01-01T00:01:00Z"},
    {"event_type": "FAILURE_REPAIR", "category": "worker_env", "evidence": "codex lacked claude", "repair": "exec-check", "timestamp": "2026-01-01T00:02:00Z"},
    {"event_type": "PROCEDURE_OUTCOME", "procedure": "p1", "outcome": "success", "timestamp": "2026-01-01T00:03:00Z"},
    {"kind": "intervention", "category": "execution_failure_correction", "reason": "stopped at partial", "timestamp": "2026-01-01T00:04:00Z"},
    {"kind": "lesson", "id": "L-1", "trigger": "rename", "action": "search whole repo", "timestamp": "2026-01-01T00:05:00Z"},
    {"kind": "lesson_use", "lesson_id": "L-1", "outcome": "no_effect", "timestamp": "2026-01-01T00:06:00Z"},
]
MISSION = {"id": "MISSION-901", "title": "Fixture mission", "status": "RESOLVED", "verdict": "FALSIFIED", "hypothesis": "HYPOTHESIS-901 evens beat irregular",
           "falsification_criteria": "FALSIFIED if diff < 0.05", "experiment": "exact enumeration", "evidence": ["results.json"], "lessons_learned": ["LESSON-TEXT-901"],
           "design_constraints": ["LIMITATION-901 symmetric rhythms"], "related_missions": {"MISSION-902": "sibling"}, "created_at": "x", "updated_at": "y", "execution_surface": "local",
           "scores": {}, "model_routing": {}, "tags": []}


def make_repo(publish=True):
    d = Path(tempfile.mkdtemp()); origin = Path(tempfile.mkdtemp())
    git(origin, "init", "-q", "--bare", "-b", "main")
    git(d, "init", "-q", "-b", "main")
    files = {
        "chronicle/entries/001_first.md": "# Entry 1\nCHRONICLE-TEXT\n",
        "LEARNING/missions/MISSION-901.json": json.dumps(MISSION),
        "LEARNING/experiments/mission-901/README.md": "# Result\nFINDING-901: even rhythms are worse. LIMITATION-901 noted.\n",
        "LEARNING/experiments/mission-901/results.json": json.dumps({"x": 1}),
        "LEARNING/experiments/mission-901/run.py": "print('hi')\n",
        "LEARNING/topics/t/README.md": "# Topic\nTOPIC-TEXT\n",
        ".agent/learning/events.jsonl": "\n".join(json.dumps(e) for e in EVENTS) + "\n",
        ".agent/learning/learning_package_snapshot.md": "# Seal\nSEALED\n",
        ".agent/learning/usage_latest.json": "{}",
        "context/soul.md": "# Soul\nIDENTITY\n", "context/user.md": "# User\nPREF\n", "context/memory.md": "# Memory\nCURATED\n",
        "protocols/001_p.md": "# P\nPROTOCOL\n", "AGENTS.md": "# A\n", "START_HERE.md": "# S\n", "README.md": "# R\n", "architecture.md": "# Arch\n", "INSTALL.md": "# I\n",
        "LEARNING/missions/mission.schema.json": "{}",
        "plugins/sanctuary-core/.apm/skills/x/SKILL.md": "# Skill\nSKILL-DOC\n", "plugins/sanctuary-core/.apm/skills/x/scripts/a.py": "x=1\n",
    }
    for p, c in files.items():
        f = d / p; f.parent.mkdir(parents=True, exist_ok=True); f.write_text(c)
    git(d, "add", "-A"); git(d, "commit", "-q", "-m", "init")
    git(d, "remote", "add", "origin", str(origin))
    if publish:
        git(d, "push", "-q", "-u", "origin", "main")
    return d


def ids(d):
    return [json.loads(l)["id"] for l in (d / S.DATASET).read_text().splitlines() if l.strip()]


def text_of(d, rid):
    return next(json.loads(l)["content"] for l in (d / S.DATASET).read_text().splitlines() if l.strip() and json.loads(l)["id"] == rid)


def alltext(d):
    return "\n".join(text_of(d, i) for i in ids(d))


class Export(unittest.TestCase):
    def test_critical_sources_outside_the_chronicle_are_preserved_with_their_substance(self):
        d = make_repo(); m = S.export(d)
        self.assertTrue(m["summary"]["required_ok"], m["summary"]["problems"])
        for needle in ("CHRONICLE-TEXT", "FINDING-901", "LIMITATION-901", "HYPOTHESIS-901", "FALSIFIED", "LESSON-TEXT-901", "HANDOFF-TEXT-901", "codex lacked claude",
                       "stopped at partial", "search whole repo", "no_effect", "TOPIC-TEXT", "SEALED", "IDENTITY", "CURATED", "PROTOCOL", "SKILL-DOC"):
            self.assertIn(needle, alltext(d), needle)
        self.assertTrue(TE.verify_dataset(d / S.DATASET)[0])

    def test_supporting_artifacts_are_stable_references_not_scratch_paths(self):
        d = make_repo(); S.export(d)
        idx = next(text_of(d, i) for i in ids(d) if i.startswith("artifacts__"))
        h = hashlib.sha256((d / "LEARNING/experiments/mission-901/results.json").read_bytes()).hexdigest()
        self.assertIn(h, idx); self.assertIn(git(d, "rev-parse", "HEAD").strip(), idx); self.assertIn("| True |", idx)
        self.assertNotIn(str(d), idx)

    def test_unpublished_artifact_commit_makes_coverage_incomplete_not_silent(self):
        d = make_repo(publish=False); m = S.export(d)
        self.assertFalse(m["summary"]["required_ok"]); self.assertTrue(any("not yet on origin/main" in p for p in m["summary"]["problems"]))

    def test_repeated_export_is_idempotent_and_changes_only_what_changed(self):
        d = make_repo(); S.export(d); first = (d / S.DATASET).read_bytes(); n = len(ids(d))
        m2 = S.export(d)
        self.assertEqual((d / S.DATASET).read_bytes(), first)                        # byte-identical: no timestamp churn
        self.assertEqual(len(ids(d)), len(set(ids(d)))); self.assertEqual(len(ids(d)), n)
        acts = {a for e in m2["sources"] for a in (e.get("action") or {})}
        self.assertTrue(acts <= {"unchanged"}, acts)
        (d / "LEARNING/topics/t/README.md").write_text("# Topic\nTOPIC REVISED\n"); git(d, "commit", "-qam", "edit"); git(d, "push", "-q")
        m3 = S.export(d)
        self.assertIn("REVISED", text_of(d, "LEARNING_topics_t_README")); self.assertEqual(len(ids(d)), n)
        changed = [e["path"] for e in m3["sources"] if (e.get("action") or {}).get("updated")]
        self.assertEqual(changed, ["LEARNING/topics/t/README.md"])

    def test_malformed_and_unsupported_required_sources_fail_loudly(self):
        d = make_repo()
        (d / "LEARNING/missions/MISSION-902.json").write_text("{not json"); (d / "chronicle/entries/002_bin.md").write_bytes(b"\xff\xfe\x00bad")
        git(d, "add", "-A"); git(d, "commit", "-qm", "bad"); git(d, "push", "-q")
        m = S.export(d); st = {e["path"]: e for e in m["sources"]}
        self.assertEqual(st["LEARNING/missions/MISSION-902.json"]["status"], "failed"); self.assertEqual(st["chronicle/entries/002_bin.md"]["status"], "failed")
        self.assertFalse(m["summary"]["required_ok"])
        self.assertEqual(S.report(S.check(d)), 2)                                     # SOUL PARTIAL

    def test_secrets_and_intentional_exclusions(self):
        d = make_repo()
        (d / "LEARNING/topics/t/leak.md").write_text("token sk-" + "a" * 30 + "\n"); (d / ".env").write_text("X=1\n"); (d / "context").mkdir(exist_ok=True); (d / "context/credentials.json").write_text("{}")
        git(d, "add", "-A"); git(d, "commit", "-qm", "s"); git(d, "push", "-q")
        m = S.export(d); st = {e["path"]: e for e in m["sources"]}
        self.assertEqual(st["LEARNING/topics/t/leak.md"]["status"], "excluded_sensitive")
        self.assertEqual(st[".agent/learning/usage_latest.json"]["status"], "excluded"); self.assertIn("regenerated", st[".agent/learning/usage_latest.json"]["reason"])
        self.assertEqual(st["context/credentials.json"]["status"], "excluded")
        self.assertNotIn("a" * 30, alltext(d))
        self.assertFalse(m["summary"]["required_ok"])                                 # a required knowledge source was withheld for safety: stated, not silent

    def test_sensitive_event_line_is_skipped_and_reported(self):
        d = make_repo()
        with (d / ".agent/learning/events.jsonl").open("a") as f:
            f.write(json.dumps({"event_type": "CHECKIN", "handoff": "key sk-" + "b" * 30}) + "\n")
        git(d, "commit", "-qam", "e"); git(d, "push", "-q"); m = S.export(d)
        ev = next(e for e in m["sources"] if e["category"] == "event_ledger")
        self.assertEqual(ev["sensitive_events_skipped"], 1); self.assertFalse(m["summary"]["required_ok"])
        self.assertNotIn("b" * 30, alltext(d))

    def test_uncategorized_files_under_watched_roots_are_listed(self):
        d = make_repo(); (d / "LEARNING/odd.xyz").write_text("?"); git(d, "add", "-A"); git(d, "commit", "-qm", "u"); git(d, "push", "-q")
        self.assertIn("LEARNING/odd.xyz", S.export(d)["uncategorized_under_watched_roots"])


class Recovery(unittest.TestCase):
    def test_soul_index_links_mission_verdict_records_and_artifacts(self):
        d = make_repo(); S.export(d)
        idx = text_of(d, "soul_index")
        self.assertIn("MISSION-901", idx); self.assertIn("FALSIFIED", idx); self.assertIn("mission__MISSION-901", idx)
        self.assertIn("LEARNING_experiments_mission-901_README", idx); self.assertIn("artifacts__LEARNING_experiments_mission_901", idx)
        self.assertEqual(S.export(d)["index_record_id"], "soul_index")

    def test_closure_reports_remote_unverified_and_status_line_is_honest(self):
        d = make_repo()
        self.assertEqual(S.main(["closure", "--repo-root", str(d)]), 4)             # exported + complete, but nothing verified remotely
        st, msg = S.quick_status(d)
        self.assertEqual(st, "WARN"); self.assertIn("remote NOT verified", msg)
        (d / "LEARNING/topics/t/README.md").write_text("# Topic\nNEW\n"); git(d, "commit", "-qam", "c")
        self.assertIn("STALE", S.quick_status(d)[1])

    def test_manifest_dataset_integrity_is_checked(self):
        d = make_repo(); S.export(d)
        lines = (d / S.DATASET).read_text().splitlines()
        rec = json.loads(lines[0]); rec["content"] = rec["content"] + " tampered"; rec["sha256"] = hashlib.sha256(rec["content"].encode()).hexdigest()
        lines[0] = json.dumps(rec); (d / S.DATASET).write_text("\n".join(lines) + "\n")
        self.assertTrue(any("INTEGRITY" in p for p in S.check(d)["problems"]))      # schema/hash-valid, yet no longer what the manifest recorded


class Check(unittest.TestCase):
    def test_four_facts_are_separate_and_persisted_needs_a_verified_receipt(self):
        d = make_repo(); S.export(d)
        r = S.check(d)
        self.assertEqual((r["schema"], r["coverage"], r["local_export"]), ("valid", "complete", "current")); self.assertTrue(r["remote"].startswith("NOT VERIFIED"))
        self.assertEqual(S.report(r), 4)
        ds = hashlib.sha256((d / S.DATASET).read_bytes()).hexdigest()
        SYNC.write_receipt(d, "o/r", {"verified": True, "revision": "abcdef1234567890", "local_dataset_sha256": ds, "local_manifest_sha256": S.sha((d / S.MANIFEST).read_bytes())}, True)
        self.assertEqual(S.report(S.check(d)), 0)
        (d / "LEARNING/topics/t/README.md").write_text("# Topic\nCHANGED\n"); git(d, "commit", "-qam", "c")
        r2 = S.check(d); self.assertEqual(r2["local_export"], "STALE"); self.assertEqual(S.report(r2), 3)    # changed source -> not current, not "persisted"

    def test_committing_the_sync_receipt_does_not_make_the_export_stale(self):
        d = make_repo(); S.export(d)
        ds = hashlib.sha256((d / S.DATASET).read_bytes()).hexdigest()
        SYNC.write_receipt(d, "o/r", {"verified": True, "revision": "abcdef1234567890", "local_dataset_sha256": ds, "local_manifest_sha256": S.sha((d / S.MANIFEST).read_bytes())}, True)
        git(d, "add", "-A"); git(d, "commit", "-qm", "receipt"); git(d, "push", "-q")
        self.assertEqual(S.report(S.check(d)), 0)                                     # still SOUL PERSISTED after the receipt is committed

    def test_changing_excluded_code_does_not_stale_the_export_but_changing_a_preserved_doc_does(self):
        d = make_repo(); S.export(d)
        (d / "plugins/sanctuary-core/.apm/skills/x/scripts/a.py").write_text("x=2\n"); git(d, "commit", "-qam", "code"); git(d, "push", "-q")
        self.assertEqual(S.check(d)["local_export"], "current")
        (d / "plugins/sanctuary-core/.apm/skills/x/SKILL.md").write_text("# Skill\nCHANGED DOC\n"); git(d, "commit", "-qam", "doc"); git(d, "push", "-q")
        self.assertEqual(S.check(d)["local_export"], "STALE")

    def test_receipt_for_an_older_dataset_does_not_verify_the_new_one(self):
        d = make_repo(); S.export(d)
        SYNC.write_receipt(d, "o/r", {"verified": True, "revision": "r1", "local_dataset_sha256": "0" * 64}, True)
        self.assertTrue(S.check(d)["remote"].startswith("NOT VERIFIED"))

    def test_remote_verification_compares_hashes(self):
        d = make_repo(); S.export(d)
        ds, mf = d / S.DATASET, d / S.MANIFEST
        good = {"data/soul_traces.jsonl": ds.read_bytes(), "metadata/coverage_manifest.json": mf.read_bytes()}
        self.assertTrue(SYNC.verify_remote("o/r", ds, mf, "rev1", fetch=lambda rid, fn, rev: good[fn])["verified"])
        bad = SYNC.verify_remote("o/r", ds, mf, "rev1", fetch=lambda rid, fn, rev: b"different")
        self.assertFalse(bad["verified"]); self.assertEqual(len(bad["problems"]), 2)
        self.assertFalse(SYNC.verify_remote("o/r", ds, mf, "rev1", fetch=lambda *a: (_ for _ in ()).throw(OSError("net down")))["verified"])

    def test_preservation_purpose_reference_is_covered_under_skill_documentation(self):
        d = make_repo()
        purpose_doc = d / "plugins/sanctuary-core/.apm/skills/preserving-soul-traces/references/preservation-purpose.md"
        purpose_doc.parent.mkdir(parents=True, exist_ok=True)
        purpose_doc.write_text("# Preservation Purpose\n\nPreserve intellectual inheritance faithfully.\n")
        git(d, "add", "-A"); git(d, "commit", "-qm", "add purpose ref"); git(d, "push", "-q")
        policy = S.load_policy()
        rule = S.classify("plugins/sanctuary-core/.apm/skills/preserving-soul-traces/references/preservation-purpose.md", policy)
        self.assertEqual(rule["name"], "skill_documentation")
        manifest = S.export(d)
        src = [s for s in manifest["sources"] if "preservation-purpose.md" in s["path"]]
        self.assertEqual(len(src), 1)
        self.assertEqual(src[0]["status"], "included")
        recs = S.load_dataset(d)
        rec = [r for r in recs.values() if "preservation-purpose.md" in r["source_file"]]
        self.assertEqual(len(rec), 1)
        self.assertIn("Preserve intellectual inheritance faithfully.", rec[0]["content"])


if __name__ == "__main__":
    unittest.main()

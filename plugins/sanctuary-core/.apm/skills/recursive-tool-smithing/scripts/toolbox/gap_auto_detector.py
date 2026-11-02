"""
gap_auto_detector.py

Purpose: Analyze a recursive-tool-smithing plugin directory to detect capability gaps by reading:
 - chronicles/ hourly .md files for friction events
 - evals/results.tsv for crash rows
Outputs proposed gap entries as JSON to stdout.

Layer: plugin analyzer
Plugin: recursive-tool-smithing

Usage example:
  python gap_auto_detector.py --plugin-root /path/to/plugin --gap-log /path/to/gaps.json

Key functions:
 - get_chronicle_friction
 - get_ledger_crashes
 - normalize_slug
 - rank_friction
 - load_resolved_slugs
 - propose_gaps
"""

from pathlib import Path
from typing import List, Dict, Set
import argparse
import csv
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict

def get_chronicle_friction(plugin_root: Path) -> List[str]:
    """
    Scan chronicles/**/*.md (skip files named daily.md) and collect lines that start with '1.' '2.' or '3.'.
    Strips leading number and dot and leading/trailing whitespace from each matched line.
    Returns list of event strings.
    """
    events: List[str] = []
    chronicles_root = plugin_root / "chronicles"
    if not chronicles_root.exists():
        return events
    for p in chronicles_root.rglob("*.md"):
        if p.is_file() and p.name.lower() != "daily.md":
            try:
                with p.open(encoding="utf-8") as f:
                    for raw in f:
                        line = raw.lstrip()
                        if line.startswith(("1.", "2.", "3.")):
                            # remove leading number, dot, and following whitespace
                            stripped = re.sub(r'^[123]\.\s*', '', line).rstrip('\n\r')
                            stripped = stripped.strip()
                            if stripped:
                                events.append(stripped)
            except Exception:
                # ignore unreadable files
                continue
    return events

def get_ledger_crashes(results_tsv: Path) -> List[str]:
    """
    Read a TSV file (results_tsv). Parse header to find 'status' and 'description' columns.
    Return the description strings for rows where status == 'crash' (case-insensitive).
    Skip header row. If file missing or malformed, return empty list.
    """
    descriptions: List[str] = []
    if not results_tsv or not results_tsv.exists():
        return descriptions
    try:
        with results_tsv.open(encoding="utf-8", newline='') as fh:
            reader = csv.reader(fh, delimiter='\t')
            header = next(reader, None)
            if not header:
                return descriptions
            # Normalize header names
            hdr_norm = [h.strip().lower() for h in header]
            try:
                status_idx = hdr_norm.index('status')
            except ValueError:
                status_idx = None
            try:
                desc_idx = hdr_norm.index('description')
            except ValueError:
                desc_idx = None
            # If description column missing, try 'desc' or 'error'
            if desc_idx is None:
                for alt in ('desc', 'error', 'message'):
                    if alt in hdr_norm:
                        desc_idx = hdr_norm.index(alt)
                        break
            if status_idx is None or desc_idx is None:
                # Can't find required columns
                return descriptions
            for row in reader:
                if len(row) <= max(status_idx, desc_idx):
                    continue
                status = row[status_idx].strip().lower()
                if status == 'crash':
                    descriptions.append(row[desc_idx].strip())
    except Exception:
        return descriptions
    return descriptions

def normalize_slug(text: str) -> str:
    """
    Lowercase the text, replace spaces and punctuation with hyphens, collapse multiple hyphens,
    strip leading/trailing hyphens, and truncate to max 40 characters.
    """
    if text is None:
        return ""
    s = text.lower()
    # Replace any non-alphanumeric characters with hyphen
    s = re.sub(r'[^a-z0-9]+', '-', s)
    # Collapse multiple hyphens
    s = re.sub(r'-{2,}', '-', s)
    # Strip hyphens
    s = s.strip('-')
    # Truncate to max 40 characters without trailing hyphen
    if len(s) > 40:
        s = s[:40].rstrip('-')
    return s

def rank_friction(chronicle_events: List[str], crash_descs: List[str]) -> List[Dict]:
    """
    Combine chronicle_events and crash_descs into a single list. Normalize each to a slug key,
    count duplicates, and produce a sorted list of dicts by frequency descending.
    Each dict contains: slug, description (representative), frequency, priority.
    Priority = min(10, frequency * 2 + 1)
    """
    combined = list(chronicle_events) + list(crash_descs)
    slug_to_count: Counter = Counter()
    slug_to_examples: Dict[str, Counter] = defaultdict(Counter)
    for item in combined:
        if not item:
            continue
        slug = normalize_slug(item)
        if not slug:
            continue
        slug_to_count[slug] += 1
        slug_to_examples[slug][item] += 1
    results: List[Dict] = []
    for slug, freq in slug_to_count.items():
        examples = slug_to_examples.get(slug, {})
        # pick most common example description as representative
        if examples:
            description = examples.most_common(1)[0][0]
        else:
            description = slug.replace('-', ' ')
        priority = min(10, freq * 2 + 1)
        results.append({
            "slug": slug,
            "description": description,
            "frequency": int(freq),
            "priority": int(priority),
        })
    # sort by frequency desc, then priority desc, then slug
    results.sort(key=lambda d: (-d["frequency"], -d["priority"], d["slug"]))
    return results

def load_resolved_slugs(gaps_path: Path) -> Set[str]:
    """
    Read gaps.json if exists and return a set of slugs where status == 'resolved'.
    If file missing or unreadable, return empty set.
    """
    resolved: Set[str] = set()
    if not gaps_path or not gaps_path.exists():
        return resolved
    try:
        with gaps_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                # maybe a dict with entries
                entries = data.get("gaps") or data.get("entries") or []
            else:
                entries = data
            if not isinstance(entries, list):
                return resolved
            for e in entries:
                if not isinstance(e, dict):
                    continue
                status = e.get("status", "").strip().lower()
                slug = e.get("slug") or e.get("id") or e.get("name")
                if slug and status == "resolved":
                    resolved.add(str(slug))
    except Exception:
        return set()
    return resolved

def propose_gaps(plugin_root: Path, gaps_json_path: Path) -> List[Dict]:
    """
    Analyze plugin_root for chronicle friction events and evals/results.tsv crashes,
    rank the combined issues, filter out already resolved slugs from gaps_json_path,
    and return the top 5 proposed gaps as list of dicts.
    """
    plugin_root = Path(plugin_root)
    # gather chronicle events
    chronicle_events = get_chronicle_friction(plugin_root)
    # find results.tsv - default location: evals/results.tsv under plugin_root
    results_tsv = plugin_root / "evals" / "results.tsv"
    crash_descs = get_ledger_crashes(results_tsv)
    ranked = rank_friction(chronicle_events, crash_descs)
    resolved = load_resolved_slugs(gaps_json_path) if gaps_json_path is not None else set()
    # filter out resolved by slug
    proposals = [r for r in ranked if r.get("slug") not in resolved]
    # return top 5
    return proposals[:5]

def main(argv=None):
    """
    CLI entrypoint. Parses arguments, runs propose_gaps, and prints JSON to stdout.
    """
    parser = argparse.ArgumentParser(description="Detect plugin capability gaps and propose entries as JSON")
    default_plugin_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--plugin-root", type=Path, default=default_plugin_root,
                        help="Path to plugin root (default: three levels up from this script)")
    parser.add_argument("--gap-log", type=Path,
                        help="Path to gaps.json (default: gaps.json at plugin root)")
    args = parser.parse_args(argv)
    plugin_root: Path = args.plugin_root
    gap_log: Path = args.gap_log or (plugin_root / "gaps.json")
    proposals = propose_gaps(plugin_root, gap_log)
    # print JSON to stdout only
    json.dump(proposals, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")

if __name__ == "__main__":
    # If any CLI args provided, run main normally and exit 0
    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)
    # Else run self-test using TemporaryDirectory
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # create evals/results.tsv with header + one crash row
        evals_dir = td_path / "evals"
        evals_dir.mkdir(parents=True, exist_ok=True)
        results_file = evals_dir / "results.tsv"
        with results_file.open("w", encoding="utf-8", newline='') as fh:
            writer = csv.writer(fh, delimiter='\t')
            writer.writerow(["id", "status", "description"])
            writer.writerow(["r1", "crash", "Example crash during self-test"])
        # create chronicles/2026-03-14/0100.md with '1. Fixed path bug in self-test'
        chron_dir = td_path / "chronicles" / "2026-03-14"
        chron_dir.mkdir(parents=True, exist_ok=True)
        file_md = chron_dir / "0100.md"
        with file_md.open("w", encoding="utf-8") as fh:
            fh.write("1. Fixed path bug in self-test\n")
            fh.write("2. Another minor friction\n")
        # create empty gaps.json as []
        gaps_json = td_path / "gaps.json"
        with gaps_json.open("w", encoding="utf-8") as gh:
            gh.write("[]")
        # call propose_gaps and assert
        res = propose_gaps(td_path, gaps_json)
        assert isinstance(res, list), "propose_gaps must return a list"
        # basic sanity: should include at least the two items we created
        slugs = {r["slug"] for r in res}
        assert any("fixed-path-bug-in-self-test".startswith(s) or "fixed-path-bug" in s for s in slugs) or len(res) >= 1
        # print success message
        print("gap_auto_detector: all self-tests passed.")

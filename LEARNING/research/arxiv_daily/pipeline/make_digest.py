#!/usr/bin/env python3
"""Build today's digest from incoming.json: digests/YYYY-MM-DD.md (archive) and latest.md (copy).

Deterministic, no model: summaries are the first two sentences of each abstract.
"""
import json
import re
import time
from pathlib import Path

DOSSIER = Path(__file__).resolve().parents[1]


def two_sentences(text: str) -> str:
    return " ".join(re.split(r"(?<=[.!?])\s+", text.strip())[:2])


def build(papers: list[dict], stamp: str) -> str:
    lines = ["# arXiv Daily Digest", "", f"Generated {stamp} from the {len(papers)} newest cs.AI submissions.", "",
             "## Overview", "", "Newest titles: " + "; ".join(p["title"] for p in papers) + ".", "", "## Papers", ""]
    for i, p in enumerate(papers, 1):
        lines += [f"### {i}. {p['title']}", "", f"- Link: {p['id']}", f"- Summary: {two_sentences(p['summary'])}", ""]
    lines += ["## Candidate Mission Ideas", "",
              "Unstarted seeds generated from titles; review before registering anything.", ""]
    for p in papers[:2]:
        lines.append(f"- Test the central claim of \"{p['title']}\" ({p['id']}) on a small local reproduction, and record whether it holds.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    papers = json.loads((DOSSIER / "incoming.json").read_text())
    text = build(papers, time.strftime("%Y-%m-%d %H:%M %Z"))
    day = time.strftime("%Y-%m-%d")
    (DOSSIER / "digests").mkdir(exist_ok=True)
    (DOSSIER / "digests" / f"{day}.md").write_text(text)
    (DOSSIER / "latest.md").write_text(text)
    print(f"wrote digests/{day}.md and latest.md")

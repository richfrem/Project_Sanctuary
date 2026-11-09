#!/usr/bin/env python3
"""Fetch the 5 newest arXiv cs.AI papers (stdlib only) into the dossier's incoming.json."""
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DOSSIER = Path(__file__).resolve().parents[1]

URL = ("http://export.arxiv.org/api/query?search_query=cat:cs.AI"
       "&sortBy=submittedDate&sortOrder=descending&max_results=5")
NS = {"a": "http://www.w3.org/2005/Atom"}


def fetch() -> list[dict]:
    req = urllib.request.Request(URL, headers={"User-Agent": "sanctuary-arxiv-digest/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        root = ET.fromstring(r.read())
    out = []
    for e in root.findall("a:entry", NS):
        clean = lambda t: " ".join((e.findtext(t, "", NS) or "").split())
        out.append({"id": clean("a:id"), "title": clean("a:title"), "summary": clean("a:summary")})
    return out


if __name__ == "__main__":
    papers = fetch()
    (DOSSIER / "incoming.json").write_text(json.dumps(papers, indent=2) + "\n")
    print(f"saved {len(papers)} papers to {DOSSIER / 'incoming.json'}")

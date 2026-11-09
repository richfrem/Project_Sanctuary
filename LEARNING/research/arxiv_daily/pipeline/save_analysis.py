#!/usr/bin/env python3
"""Save the AI analysis step's output: analysis/YYYY-MM-DD.md and new seeds appended to seeds.jsonl.

Usage: save_analysis.py <analysis_output_file>
Fails (exit 1) when the required sections are missing, so the conductor retries or stops the run.
"""
import json
import re
import sys
import time
from pathlib import Path

DOSSIER = Path(__file__).resolve().parents[1]
REQUIRED = ["## Relevance to existing missions", "## Impact on the project", "## Improvement ideas", "## New mission seeds"]


def clean(text: str) -> str:
    """Drop CLI preamble before the first markdown heading."""
    i = text.find("# ")
    return text[i:] if i >= 0 else text


def seeds_from(text: str) -> list[dict]:
    m = re.search(r"```jsonl\s*\n(.*?)```", text, re.S)
    out = []
    for line in (m.group(1).splitlines() if m else []):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("question"):
            out.append({k: d.get(k) for k in ("paper", "question", "first_step", "related_missions")})
    return out


def main(src: str) -> int:
    text = clean(Path(src).read_text(errors="replace"))
    missing = [h for h in REQUIRED if h not in text]
    if missing:
        print("analysis is missing sections: " + ", ".join(missing), file=sys.stderr)
        return 1
    day = time.strftime("%Y-%m-%d")
    (DOSSIER / "analysis").mkdir(exist_ok=True)
    (DOSSIER / "analysis" / f"{day}.md").write_text(text.rstrip() + "\n")
    seeds = seeds_from(text)
    with open(DOSSIER / "seeds.jsonl", "a") as f:
        for s in seeds:
            f.write(json.dumps({"date": day, **s, "status": "unreviewed", "source": f"analysis/{day}.md"}) + "\n")
    print(f"wrote analysis/{day}.md and {len(seeds)} seed(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

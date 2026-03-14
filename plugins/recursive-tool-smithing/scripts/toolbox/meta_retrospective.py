from pathlib import Path
import argparse
from datetime import datetime, timezone
from typing import Optional

def read_last_non_header_line(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    filtered = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
    return filtered[-1] if filtered else None

def generate_suggestion(friction: str) -> str:
    s = friction.lower()
    if any(k in s for k in ("slow", "latency", "lag", "sluggish")):
        return "Measure and optimize the slow step (profiling, caching, or parallelism)."
    if any(k in s for k in ("manual", "copy", "paste", "human")):
        return "Automate this manual step (script, CI job, or tool integration)."
    if any(k in s for k in ("unclear", "confusing", "ambiguous", "documentation")):
        return "Clarify docs and add examples or error messages."
    if any(k in s for k in ("error", "fail", "crash", "bug")):
        return "Add robust error handling and reproducible test to isolate the failure."
    # fallback: concise imperative suggestion derived from friction
    short = friction.strip()
    if len(short) > 120:
        short = short[:117].rsplit(" ", 1)[0] + "..."
    return f"Refine process: {short}"

def append_entry(log_path: Path, timestamp: str, gap: str, status: str, friction: str, suggestion: str, context: Optional[str]) -> None:
    entry = f"- {timestamp} | gap: {gap} | status: {status} | friction: {friction} | suggestion: {suggestion}"
    if context:
        entry += f" | context: {context}"
    entry += "\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)

def main() -> None:
    p = argparse.ArgumentParser(description="Meta retrospective logger")
    p.add_argument("--status", required=True, choices=["keep","discard","crash"])
    p.add_argument("--gap", required=True)
    p.add_argument("--friction", required=True)
    args = p.parse_args()
    results_path = Path("skills/recursive-tool-smithing/evals/results.tsv")
    context = read_last_non_header_line(results_path)
    friction_clean = " ".join(args.friction.splitlines()).strip()
    suggestion = generate_suggestion(friction_clean)
    ts = datetime.now(timezone.utc).isoformat()
    log_path = Path("planning/friction_log.md")
    append_entry(log_path, ts, args.gap, args.status, friction_clean, suggestion, context)
    print(f"Appended retrospective entry to {log_path}") 

if __name__ == "__main__":
    if len(__import__("sys").argv) == 1:
        print("Self-test: meta_retrospective ready. Pass --status --gap --friction to log.")
        raise SystemExit(0)
    main()

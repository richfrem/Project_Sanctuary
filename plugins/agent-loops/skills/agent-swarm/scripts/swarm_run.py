#!/usr/bin/env python3
"""
swarm_run.py
============

Generic parallel Claude CLI executor driven by a self-contained "job file".
A job file is a Markdown file with YAML frontmatter encoding all configuration.
The Markdown body is the prompt sent to Claude for every input file.

Usage:
    python3 swarm_run.py --job <JOB_FILE> [--dir DIR | --files-from LIST | --files F ...]

Overrides (all optional — job file is the source of truth):
    --model haiku|sonnet|opus   Override the job model.
    --workers N                 Override concurrency.
    --var KEY=VALUE             Inject or override a template variable.
    --dir / --files-from / --files   Override the input source.

Job file format (YAML frontmatter + Markdown prompt):
----------------------------------------------------
---
model: haiku
workers: 10
timeout: 120
ext: [".md"]

# Shell command run after each file. Placeholders:
#   {file}           — relative path of input file
#   {output}         — Claude output, shell-safe single-quoted
#   {output_raw}     — Claude output unquoted (use carefully)
#   {basename}       — filename without dir or extension
#   {<var>}          — any key defined in vars: below
post_cmd: >
  python3 plugins/rlm-factory/skills/rlm-curator/scripts/inject_summary.py
  --profile {profile} --file {file} --summary {output}

# Optional: save each output as <basename>.txt in this directory
output_dir: null

vars:
  profile: project
---

Your prompt text goes here. This is sent verbatim to Claude for every input file.
Claude receives the file content via stdin.
----------------------------------------------------
"""

import re
import sys
import argparse
import subprocess
import concurrent.futures
from pathlib import Path

try:
    import yaml
except ImportError:
    print("❌ PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


# ─── Job File Parsing ─────────────────────────────────────────────────────────

def load_job(job_path: Path) -> tuple[dict, str]:
    """
    Parse a job file with YAML frontmatter.
    Returns (config_dict, prompt_text).
    """
    text = job_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text.strip()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip()
    config = yaml.safe_load(parts[1]) or {}
    prompt = parts[2].strip()
    return config, prompt


# ─── File Resolution ──────────────────────────────────────────────────────────

def get_dir_files(directory: Path, extensions: list[str]) -> list[str]:
    root = Path.cwd().resolve()
    abs_dir = directory.resolve()
    exts = set(e if e.startswith(".") else f".{e}" for e in extensions)
    return [
        str(f.relative_to(root))
        for f in sorted(abs_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in exts and not f.name.startswith(".")
    ]


def parse_task_list(task_md: Path) -> list[str]:
    paths = []
    for line in task_md.read_text(encoding="utf-8").splitlines():
        m = re.match(r"- \[ \] `(.+)`", line)
        if m:
            paths.append(m.group(1))
    return paths


def parse_bundle(bundle_path: Path) -> list[str]:
    """Read a context-bundler manifest (JSON or YAML) and return file paths.

    Supports two formats:
      JSON: {"files": [{"path": "...", "note": "..."}, ...]}
      YAML: files:\n  - path: ...\n    note: ...
    Also accepts a bare list of path strings.
    """
    import json
    text = bundle_path.read_text(encoding="utf-8")
    # Try JSON first
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        data = yaml.safe_load(text)

    if data is None:
        return []

    # Normalise to a list of path strings
    if isinstance(data, dict):
        data = data.get("files", [])
    if isinstance(data, list):
        paths = []
        for item in data:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict):
                p = item.get("path") or item.get("file")
                if p:
                    paths.append(str(p))
        return paths
    return []


# ─── Shell Helpers ────────────────────────────────────────────────────────────

def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


# ─── Worker ───────────────────────────────────────────────────────────────────

def run_worker(
    file_path: str,
    prompt: str,
    model: str,
    post_cmd_template: str | None,
    output_dir: Path | None,
    user_vars: dict,
    timeout: int,
) -> tuple[str, bool, str]:
    # Read source file
    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        return (file_path, False, f"Read error: {e}")

    # Call Claude
    result = subprocess.run(
        ["claude", "--model", model, "-p", prompt],
        input=content, text=True, capture_output=True, timeout=timeout,
    )
    if result.returncode != 0:
        combined = ((result.stderr or "") + (result.stdout or "")).strip()
        # Surface rate limit clearly instead of burying in 'unknown'
        if "hit your limit" in combined or "rate limit" in combined.lower():
            return (file_path, False, f"RATE_LIMIT: {combined[:200]}")
        err = combined[:300] or "unknown"
        return (file_path, False, f"Claude: {err}")

    output = result.stdout.strip()
    if not output:
        return (file_path, False, "Empty output")

    # Save to output_dir
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / (Path(file_path).stem + ".txt")).write_text(output, encoding="utf-8")

    # Run post-command
    if post_cmd_template:
        subs = {
            "file":       file_path,
            "output":     shell_quote(output),
            "output_raw": output,
            "basename":   Path(file_path).stem,
            **user_vars,
        }
        try:
            cmd = post_cmd_template.format_map(subs)
        except KeyError as e:
            return (file_path, False, f"Missing template var: {e}")

        pr = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if pr.returncode != 0:
            err = (pr.stderr or pr.stdout or "unknown").strip()[:300]
            return (file_path, False, f"Post-cmd: {err}")

    return (file_path, True, (output[:80] + "...") if len(output) > 80 else output)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="swarm_run.py",
        description="Job-file-driven parallel Claude CLI executor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--job", type=Path, required=True,
        help="Path to the job file (.md with YAML frontmatter).")

    # ── Input sources (mutually exclusive) ──
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--dir", type=Path, help="Override: process all files in DIR.")
    src.add_argument("--files-from", type=Path, help="Override: task checklist (.md).")
    src.add_argument("--files", nargs="+", help="Override: explicit file list.")
    src.add_argument("--bundle", type=Path, metavar="MANIFEST",
        help="Override: context-bundler manifest file (JSON/YAML, files[].path).")

    # Optional overrides
    parser.add_argument("--model", help="Override job model (haiku/sonnet/opus).")
    parser.add_argument("--workers", type=int, help="Override job worker count.")
    parser.add_argument("--var", action="append", default=[], metavar="KEY=VALUE",
        help="Override or add a template variable. Repeatable.")
    parser.add_argument("--output-dir", type=Path,
        help="Override: save each output as <basename>.txt here.")

    args = parser.parse_args()

    # Load job
    if not args.job.exists():
        print(f"❌ Job file not found: {args.job}")
        sys.exit(1)
    config, prompt = load_job(args.job)

    if not prompt:
        print(f"❌ Job file has no prompt body: {args.job}")
        sys.exit(1)

    # Merge config → CLI overrides
    model       = args.model   or config.get("model",   "haiku")
    workers     = args.workers or config.get("workers", 10)
    timeout     = config.get("timeout", 120)
    ext         = config.get("ext", [".md"])
    post_cmd    = config.get("post_cmd")
    raw_out_dir = args.output_dir or (Path(config["output_dir"]) if config.get("output_dir") else None)
    base_vars   = config.get("vars", {}) or {}

    # --var overrides
    for item in args.var:
        if "=" not in item:
            print(f"❌ --var must be KEY=VALUE, got: {item!r}")
            sys.exit(1)
        k, v = item.split("=", 1)
        base_vars[k.strip()] = v.strip()

    # Resolve files
    if args.dir:
        files = get_dir_files(args.dir, ext)
        src_desc = str(args.dir)
    elif args.files_from:
        files = parse_task_list(args.files_from)
        src_desc = str(args.files_from)
    elif args.files:
        files = args.files
        src_desc = f"{len(args.files)} explicit files"
    elif args.bundle:
        files = parse_bundle(args.bundle)
        src_desc = f"bundle:{args.bundle}"
    elif config.get("dir"):
        files = get_dir_files(Path(config["dir"]), ext)
        src_desc = config["dir"]
    elif config.get("files_from"):
        files = parse_task_list(Path(config["files_from"]))
        src_desc = config["files_from"]
    elif config.get("bundle"):
        files = parse_bundle(Path(config["bundle"]))
        src_desc = f"bundle:{config['bundle']}"
    else:
        print("❌ No input source. Add --dir, --files-from, --files, or --bundle (or set in job YAML).")
        sys.exit(1)

    if not files:
        print("No files to process. Exiting.")
        sys.exit(0)

    print(f"🚀 Job: {args.job.name}")
    print(f"   Source:  {src_desc} ({len(files)} files)")
    print(f"   Model:   {model}  |  Workers: {workers}  |  Timeout: {timeout}s")
    if post_cmd:
        print(f"   Post:    {str(post_cmd).strip()[:90]}")
    if base_vars:
        print(f"   Vars:    {base_vars}")
    print("-" * 70)

    success, failed = 0, []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_worker, f, prompt, model,
                        post_cmd, raw_out_dir, base_vars, timeout): f
            for f in files
        }
        for future in concurrent.futures.as_completed(futures):
            fp, ok, msg = future.result()
            if ok:
                success += 1
                print(f"  ✅ {fp}")
            else:
                failed.append((fp, msg))
                print(f"  ❌ {fp}: {msg}")

    print("-" * 70)
    print(f"Done. ✅ {success} | ❌ {len(failed)}")
    if failed:
        print("\nFailed:")
        for fp, msg in failed:
            print(f"  {fp}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

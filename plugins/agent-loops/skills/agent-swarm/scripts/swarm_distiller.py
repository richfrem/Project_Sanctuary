#!/usr/bin/env python3
"""
swarm_distiller.py
==================

Purpose:
    Generic Agent Swarm orchestrator. Dispatches N parallel Claude CLI workers
    to process a set of files using a customizable prompt file, then pipes each
    file's output to an optional post-command (e.g. inject_summary.py).

    Implements the Agent Swarm pattern from:
        plugins/agent-loops/skills/agent-swarm/SKILL.md

Usage:
    python3 plugins/agent-loops/skills/agent-swarm/scripts/swarm_distiller.py \\
        --prompt-file plugins/rlm-factory/resources/prompts/rlm/rlm_summarize_general.md \\
        --profile project \\
        --dir 00_CHRONICLE/ENTRIES \\
        --post-cmd "python3 plugins/rlm-factory/skills/rlm-curator/scripts/inject_summary.py --profile {profile} --file {file} --summary {output}" \\
        --model haiku \\
        --workers 10

    Without a post-command (just generate summaries to stdout files):
        python3 ... --prompt-file my_prompt.md --dir some/directory --output-dir /tmp/summaries

Worker selection (per agent-swarm SKILL.md):
    - haiku  → fast, cheap (docs, summaries, routine tasks)  ← default
    - sonnet → balanced (code review, architecture analysis)
    - opus   → highest reasoning (complex logic, security)
"""

import sys
import shlex
import argparse
import subprocess
import concurrent.futures
from pathlib import Path

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "haiku"
DEFAULT_WORKERS = 10

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_prompt(prompt_file: Path) -> str:
    """Load a prompt string from a file, stripping YAML frontmatter if present."""
    text = prompt_file.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()


def get_dir_files(directory: Path, extensions: list[str]) -> list[str]:
    """Return relative paths for all files in a directory."""
    root = Path.cwd().resolve()
    abs_dir = directory.resolve()
    return [
        str(f.relative_to(root))
        for f in sorted(abs_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in extensions and not f.name.startswith(".")
    ]


def parse_task_list(task_md: Path) -> list[str]:
    """Extract unchecked file paths from a rlm_distill_tasks_*.md checklist."""
    import re
    paths = []
    for line in task_md.read_text(encoding="utf-8").splitlines():
        m = re.match(r"- \[ \] `(.+)`", line)
        if m:
            paths.append(m.group(1))
    return paths


def run_worker(file_path: str, prompt: str, model: str,
               post_cmd_template: str | None, output_dir: Path | None,
               profile: str | None) -> tuple[str, bool, str]:
    """
    Inner swarm worker:
    1. Reads file content.
    2. Pipes to Claude CLI with the provided prompt.
    3. Optionally writes output to a file or runs a post-command.

    Returns (file_path, success, output_or_error).
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        return (file_path, False, f"Read error: {e}")

    cmd = ["claude", "--model", model, "-p", prompt]
    result = subprocess.run(cmd, input=content, text=True, capture_output=True, timeout=120)

    if result.returncode != 0:
        err = result.stderr.strip()[:300]
        return (file_path, False, f"Claude error: {err}")

    output = result.stdout.strip()
    if not output:
        return (file_path, False, "Empty output returned by Claude")

    # Write to output_dir if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / (Path(file_path).stem + ".txt")
        out_file.write_text(output, encoding="utf-8")

    # Run post-command if specified
    if post_cmd_template:
        # Template substitutions: {file}, {output}, {profile}
        safe_output = output.replace("'", "'\\''")  # shell-escape single quotes
        post_cmd = post_cmd_template.format(
            file=file_path,
            output=safe_output,
            profile=profile or "",
        )
        post_result = subprocess.run(post_cmd, shell=True, text=True, capture_output=True)
        if post_result.returncode != 0:
            err = post_result.stderr.strip()[:300]
            return (file_path, False, f"Post-command error: {err}")

    return (file_path, True, output)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generic Agent Swarm: parallel Claude CLI workers over a file set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Prompt
    parser.add_argument(
        "--prompt-file", type=Path, required=True,
        help="Path to .md or .txt file containing the Claude prompt (YAML frontmatter is stripped)."
    )
    # File input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="Process all matching files in this directory.")
    group.add_argument("--files-from", type=Path, help="Path to rlm_distill_tasks_*.md checklist (unchecked items only).")
    group.add_argument("--files", nargs="+", help="Explicit list of file paths to process.")
    # Worker config
    parser.add_argument("--model", default=DEFAULT_MODEL,
        help=f"Claude model alias (haiku | sonnet | opus). Default: {DEFAULT_MODEL}")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel workers. Default: {DEFAULT_WORKERS}")
    parser.add_argument("--ext", nargs="+", default=[".md"],
        help="File extensions to match when using --dir (default: .md)")
    # Output
    parser.add_argument("--post-cmd", dest="post_cmd",
        help="Shell command template run after each file. Placeholders: {file}, {output}, {profile}")
    parser.add_argument("--profile",
        help="Profile name substituted into --post-cmd as {profile} (e.g. 'project')")
    parser.add_argument("--output-dir", type=Path,
        help="Optional directory to write per-file output text files.")

    args = parser.parse_args()

    # Load prompt
    if not args.prompt_file.exists():
        print(f"❌ Prompt file not found: {args.prompt_file}")
        sys.exit(1)
    prompt = load_prompt(args.prompt_file)

    # Resolve file list
    if args.dir:
        files = get_dir_files(args.dir, args.ext)
        print(f"📁 Found {len(files)} files in {args.dir}")
    elif args.files_from:
        files = parse_task_list(args.files_from)
        print(f"📋 Loaded {len(files)} unchecked files from {args.files_from}")
    else:
        files = args.files
        print(f"� Processing {len(files)} explicit files")

    if not files:
        print("No files to process. Exiting.")
        sys.exit(0)

    print(f"🚀 Swarm ready: {len(files)} files | {args.workers} workers | model: {args.model}")
    print(f"   Prompt: {args.prompt_file}")
    if args.post_cmd:
        print(f"   Post-cmd: {args.post_cmd[:80]}...")
    print("-" * 70)

    success_count = 0
    fail_count = 0
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {
            executor.submit(
                run_worker, f, prompt, args.model,
                args.post_cmd, args.output_dir, args.profile
            ): f
            for f in files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            file_path, ok, msg = future.result()
            if ok:
                success_count += 1
                print(f"  ✅ {file_path}")
            else:
                fail_count += 1
                failures.append((file_path, msg))
                print(f"  ❌ {file_path}: {msg}")

    print("-" * 70)
    print(f"Swarm complete. ✅ {success_count} succeeded | ❌ {fail_count} failed")

    if failures:
        print("\nFailed files:")
        for fp, msg in failures:
            print(f"  {fp}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

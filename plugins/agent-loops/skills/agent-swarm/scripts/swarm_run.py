#!/usr/bin/env python3
"""
swarm_run.py 2.0
================

The universal Agent Swarm executor. Features:
- Job-file driven (Markdown + YAML frontmatter).
- Checkpoint/Resume (JSON state tracking).
- Intelligent retry (Exponential backoff for rate limits).
- Verification skip (Check command short-circuits work).
- Structured logging (JSON results).
- Multiple inputs (Dir, File Lists, Task Checklists, Bundle Manifests).

USAGE:
    python3 swarm_run.py --job my_job.md [--resume] [--dry-run]
"""

import os
import re
import sys
import json
import time
import shlex
import random
import logging
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("❌ PyYAML not found. Run: pip install pyyaml")
    sys.exit(1)

# ─── LOGGING ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("swarm")

# ─── HELPERS ────────────────────────────────────────────────────────────────

def shell_quote(value: str) -> str:
    """Safe shell quoting for templates."""
    return "'" + value.replace("'", "'\\''") + "'"

def get_relative_path(path: Path) -> str:
    root = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)

# ─── FILE DISCOVERY ─────────────────────────────────────────────────────────

def resolve_files(args, config) -> list[str]:
    """Find files from CLI args or Job config."""
    exts = config.get("ext", [".md"])
    exts = set(e if e.startswith(".") else f".{e}" for e in exts)

    # 1. Explicit Files
    if args.files:
        return args.files
    
    # 2. Bundle Manifest (JSON/YAML)
    bundle_path = args.bundle or config.get("bundle")
    if bundle_path:
        bundle_path = Path(bundle_path)
        if bundle_path.exists():
            text = bundle_path.read_text()
            try:
                data = json.loads(text)
            except:
                data = yaml.safe_load(text)
            
            if isinstance(data, dict): data = data.get("files", [])
            paths = []
            for item in data:
                p = item.get("path") if isinstance(item, dict) else item
                if p: paths.append(str(p))
            return paths

    # 3. Task Checklist
    task_path = args.files_from or config.get("files_from")
    if task_path:
        task_path = Path(task_path)
        if task_path.exists():
            return [m.group(1) for m in re.finditer(r"- \[ \] `(.+)`", task_path.read_text())]

    # 4. Directory Crawl
    dir_path = args.dir or config.get("dir")
    if dir_path:
        dir_path = Path(dir_path)
        if dir_path.exists():
            return [
                get_relative_path(f)
                for f in sorted(dir_path.rglob("*"))
                if f.is_file() and f.suffix.lower() in exts and not f.name.startswith(".")
            ]

    return []

# ─── WORKER ENGINE ───────────────────────────────────────────────────────────

def execute_worker(
    file_path: str,
    prompt: str,
    model: str,
    job_config: dict,
    user_vars: dict,
    dry_run: bool
) -> dict:
    """Processes a single file. Handles retry, skip, and post-cmd."""
    start_time = time.time()
    result = {
        "file": file_path,
        "success": False,
        "output": None,
        "error": None,
        "skipped": False,
        "retries": 0
    }

    if dry_run:
        logger.info(f"  [DRY] {file_path}")
        result["success"] = True
        return result

    # 1. Skip Check
    check_cmd_tmpl = job_config.get("check_cmd")
    if check_cmd_tmpl:
        check_cmd = check_cmd_tmpl.format(file=file_path, **user_vars)
        if subprocess.run(check_cmd, shell=True, capture_output=True).returncode == 0:
            logger.info(f"  ⏩ {file_path} (already cached)")
            result["success"] = True
            result["skipped"] = True
            return result

    # 2. Read content
    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        result["error"] = f"Read error: {e}"
        return result

    # 3. LLM Call with Retry
    max_retries = job_config.get("max_retries", 3)
    backoff = 2
    
    for attempt in range(max_retries + 1):
        result["retries"] = attempt
        proc = subprocess.run(
            ["claude", "--model", model, "-p", prompt],
            input=content, text=True, capture_output=True, timeout=job_config.get("timeout", 120)
        )
        
        combined_out = (proc.stdout or "") + (proc.stderr or "")
        
        if proc.returncode == 0 and proc.stdout.strip():
            # SUCCESS
            result["output"] = proc.stdout.strip()
            result["success"] = True
            break
        
        # ERROR HANDLING
        if "hit your limit" in combined_out.lower() or "rate limit" in combined_out.lower():
            if attempt < max_retries:
                wait = (backoff ** attempt) + random.uniform(0, 1)
                logger.warning(f"  ⌛ {file_path}: Rate limit. Backing off {wait:.1f}s...")
                time.sleep(wait)
                continue
            else:
                result["error"] = "RATE_LIMIT_EXCEEDED"
                break
        
        result["error"] = combined_out.strip()[:200]
        if attempt < max_retries:
            time.sleep(1)
            continue
        break

    if not result["success"]:
        return result

    # 4. Post-Command
    post_cmd_tmpl = job_config.get("post_cmd")
    if post_cmd_tmpl and not result["skipped"]:
        subs = {
            "file": file_path,
            "output": shell_quote(result["output"]),
            "output_raw": result["output"],
            "basename": Path(file_path).stem,
            **user_vars
        }
        cmd = post_cmd_tmpl.format_map(subs)
        pr = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if pr.returncode != 0:
            result["success"] = False
            result["error"] = (pr.stderr or pr.stdout or "post-cmd failed").strip()[:300]
    
    if result["success"]:
        logger.info(f"  ✅ {file_path}")
    else:
        logger.error(f"  ❌ {file_path}: {result['error']}")

    return result

# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Professional Agent Swarm Runner")
    parser.add_argument("--job", type=Path, required=True, help="Job file (.md)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Don't call LLM")
    parser.add_argument("--dir", type=Path)
    parser.add_argument("--files-from", type=Path)
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--var", action="append", default=[])
    args = parser.parse_args()

    # Load Job
    full_text = args.job.read_text()
    if not full_text.startswith("---"): 
        print("❌ Invalid job file (no YAML frontmatter)")
        sys.exit(1)
    
    parts = full_text.split("---", 2)
    job_config = yaml.safe_load(parts[1]) or {}
    prompt = parts[2].strip()

    # Checkpoint logic
    checkpoint_path = Path(f".swarm_state_{args.job.stem}.json")
    state = {"completed": [], "failed": {}}
    if args.resume and checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text())
        logger.info(f"🔄 Resuming from checkpoint: {len(state['completed'])} items done.")

    # Overrides
    workers = args.workers or job_config.get("workers", 5)
    model = args.model or job_config.get("model", "haiku")
    user_vars = job_config.get("vars", {}) or {}
    for v in args.var:
        k, val = v.split("=", 1)
        user_vars[k.strip()] = val.strip()

    # Resolve Files
    all_files = resolve_files(args, job_config)
    pending = [f for f in all_files if f not in state["completed"]]

    if not pending:
        logger.info("✨ Everything complete. Nothing to do.")
        return

    logger.info(f"🚀 Starting Swarm: {len(pending)} pending items ({len(all_files)} total)")
    logger.info(f"   Model: {model} | Workers: {workers} | Dry-run: {args.dry_run}")
    print("-" * 70)

    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(execute_worker, f, prompt, model, job_config, user_vars, args.dry_run): f 
                for f in pending
            }
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                results.append(res)
                if res["success"]:
                    state["completed"].append(res["file"])
                else:
                    state["failed"][res["file"]] = res["error"]
                
                # Checkpoint every 5 files
                if len(results) % 5 == 0:
                    checkpoint_path.write_text(json.dumps(state, indent=2))
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Interrupted. Saving state...")
    finally:
        checkpoint_path.write_text(json.dumps(state, indent=2))
        
        # Summary
        success_count = sum(1 for r in results if r["success"])
        fail_count = sum(1 for r in results if not r["success"])
        logger.info("-" * 70)
        logger.info(f"🏁 DONE. Success: {success_count} | Failed: {fail_count}")
        
    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()

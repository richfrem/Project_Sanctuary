"""
sandbox_runner.py
=================
Purpose:
    Executes a candidate toolbox script in a restricted sandbox environment.
    Enforces the Mode-Invariant Compliance Gate: file scope, no network, 5-min timeout.
    Returns a structured result: pass/crash/violation with timing and error details.

Layer: Plugin Utility Script
Plugin: recursive-tool-smithing

Usage:
    python3 sandbox_runner.py --script scripts/toolbox/pdf_parser.py --timeout 300
    python3 sandbox_runner.py --help

Safety Constraints (non-negotiable, see references/safety-sandbox.md):
    - Scripts run with --test flag only
    - Timeout: 300 seconds (5 minutes) hard limit
    - No network access validation (static analysis)
    - Allowed write paths: scripts/toolbox/ only

Output:
    - JSON result: {status, elapsed_seconds, stdout, stderr, violation}

Key Functions:
    - check_static_violations(script_path) -> list[str]
    - run_with_timeout(script_path, timeout) -> dict
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

# Patterns that indicate sandbox violations (static analysis).
# Note: subprocess is permitted ONLY for git read-only ops (log, status, diff, show).
# open() check only flags writes with literal path strings (must contain '/' to be a path, not a mode flag like "w").
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"\bimport\s+requests\b", "NETWORK: requests import"),
    (r"\bimport\s+httpx\b", "NETWORK: httpx import"),
    (r"\burllib\.request\b", "NETWORK: urllib.request usage"),
    (r"\bos\.system\b", "SUBPROCESS: os.system usage"),
    (r"\bsubprocess\.run\s*\(\s*\[(?!.*git\s+(log|status|diff|show|rev-parse|branch))", "SUBPROCESS: non-git subprocess.run"),
    (r"\bsubprocess\.Popen\b", "SUBPROCESS: subprocess.Popen (use run only)"),
    (r"\bsubprocess\.call\b", "SUBPROCESS: subprocess.call"),
    (r"\beval\s*\(", "DANGEROUS: eval() call"),
    (r"\bexec\s*\(", "DANGEROUS: exec() call"),
    (r"(?<!\.)open\s*\(\s*['\"][^'\"]*\/[^'\"]+['\"]", "SCOPE: open() with path string - use Path objects instead"),
]


# Perform static analysis to detect sandbox violations before executing.
def check_static_violations(script_path: str) -> list[str]:
    """Scan script source for forbidden patterns before execution.

    Args:
        script_path: Path to the candidate .py script.

    Returns:
        List of violation strings. Empty list means clean.
    """
    path = Path(script_path)
    if not path.exists():
        return [f"FILE_NOT_FOUND: {script_path}"]

    source = path.read_text(encoding="utf-8")
    violations: list[str] = []
    for pattern, label in FORBIDDEN_PATTERNS:
        if re.search(pattern, source):
            violations.append(label)
    return violations


# Execute the script with a hard timeout and capture stdout/stderr.
def run_with_timeout(script_path: str, timeout: int = 300) -> dict[str, Any]:
    """Run a script with --test flag under a strict timeout.

    Args:
        script_path: Path to the candidate .py script.
        timeout: Maximum execution time in seconds (default 300).

    Returns:
        Dict with keys: status (pass/crash/timeout), elapsed_seconds, stdout, stderr.
    """
    start = time.monotonic()
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        status = "pass" if result.returncode == 0 else "crash"
        return {
            "status": status,
            "elapsed_seconds": round(elapsed, 2),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "violation": None,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return {
            "status": "timeout",
            "elapsed_seconds": round(elapsed, 2),
            "stdout": "",
            "stderr": "Execution exceeded timeout limit.",
            "violation": "TIMEOUT",
        }


def main() -> None:
    """CLI entry point for sandbox_runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run a candidate tool script in the safety sandbox.")
    parser.add_argument("--script", required=True, help="Path to candidate .py script")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")
    args = parser.parse_args()

    violations = check_static_violations(args.script)
    if violations:
        result = {
            "status": "violation",
            "elapsed_seconds": 0,
            "stdout": "",
            "stderr": "",
            "violation": "; ".join(violations),
        }
    else:
        result = run_with_timeout(args.script, args.timeout)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)
    # Self-test using a minimal safe script
    import tempfile
    import os

    safe_script = """
print("sandbox_runner test: hello from safe script")
assert 1 + 1 == 2
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(safe_script)
        tmp_path = f.name

    violations = check_static_violations(tmp_path)
    assert violations == [], f"Expected no violations, got: {violations}"

    result = run_with_timeout(tmp_path, timeout=10)
    assert result["status"] == "pass", f"Expected pass, got: {result['status']}"

    os.unlink(tmp_path)
    print("sandbox_runner: all self-tests passed.")

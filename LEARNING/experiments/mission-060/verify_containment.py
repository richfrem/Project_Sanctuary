#!/usr/bin/env python3
"""Probe macOS write containment using disposable fixtures and real subprocesses.

This checks OS write denial, not model behavior, network isolation or an AGY
worker launch. It never modifies an existing repository file.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import argparse

from common import STUDY
from worker_containment import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-id", default="containment-integrated-02")
    args = parser.parse_args()
    folder = STUDY / "verification"
    folder.mkdir(exist_ok=True)
    report = folder / f"{args.report_id}.json"
    if report.exists():
        raise SystemExit("verification already recorded; do not overwrite")
    evidence = dict(status="PREPARED", recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    prediction="deliverable writes succeed; snapshot and outside writes fail, including subprocess and symlink paths",
                    scope="same profile builder used by contained_agy.py; AGY startup verified separately",
                    write_exceptions=["deliverables", "per-worker runtime", "~/.gemini/antigravity-cli client state", "PTY devices"], checks=[])
    report.write_text(json.dumps(evidence, indent=2) + "\n")
    try:
        with tempfile.TemporaryDirectory(prefix="study060-containment-") as tmp:
            root = pathlib.Path(tmp).resolve()
            snapshot = root / "snapshot"
            output = snapshot / "deliverables"
            snapshot.mkdir()
            output.mkdir()
            protected = snapshot / "fixture.txt"
            protected.write_text("preserve\n")
            outside = root / "outside.txt"
            outside.write_text("preserve\n")
            (output / "escape").symlink_to(outside)
            profile, runtime = prepare(snapshot)
            tests = [
                ("deliverable_write", output / "allowed.txt", True),
                ("snapshot_write_denied", protected, False),
                ("outside_write_denied", outside, False),
                ("symlink_escape_denied", output / "escape", False),
                ("runtime_write", runtime / "allowed.txt", True),
                ("repository_write_denied", STUDY / ".containment-probe-must-not-exist", False),
            ]
            for name, target, allowed in tests:
                child = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable,
                                        "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('changed')", str(target)],
                                       capture_output=True, text=True, timeout=20)
                evidence["checks"].append(dict(name=name, exit_code=child.returncode,
                                              passed=(child.returncode == 0) == allowed,
                                              stderr=child.stderr[-1000:]))
            terminal = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable,
                                       "-c", "import os,pty; a,b=pty.openpty(); os.write(a,b'canary'); os.close(a); os.close(b)"],
                                      capture_output=True, text=True, timeout=20)
            evidence["checks"].append(dict(name="terminal_pty_available", exit_code=terminal.returncode,
                                          passed=terminal.returncode == 0, stderr=terminal.stderr[-1000:]))
            inherited = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable,
                                        "-c", "import subprocess,sys; sys.exit(subprocess.run([sys.executable,'-c',\"import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('child')\",sys.argv[1]]).returncode)", str(outside)],
                                       capture_output=True, text=True, timeout=20)
            evidence["checks"].append(dict(name="descendant_write_denied", exit_code=inherited.returncode,
                                          passed=inherited.returncode != 0, stderr=inherited.stderr[-1000:]))
            evidence["checks"].append(dict(name="fixtures_unchanged", passed=protected.read_text() == "preserve\n" and outside.read_text() == "preserve\n"))
            evidence["profile_text"] = profile.read_text()
            evidence["profile_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
            evidence["status"] = "PASS" if all(check["passed"] for check in evidence["checks"]) else "FAIL"
    except Exception as error:
        evidence["status"] = "BLOCKED"
        evidence["error"] = str(error)
    finally:
        report.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(dict(status=evidence["status"], checks=len(evidence["checks"]),
                          report=str(report), error=evidence.get("error"))))
    if evidence["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

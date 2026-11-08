#!/usr/bin/env python3
"""Create/check frozen inputs; prospective execution requires origin/main parity."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time
import sys

STUDY = pathlib.Path(__file__).resolve().parent
REPO = STUDY.parents[2]
STATIC = ("protocol.md", "thresholds.json", "pool-base.json", "formula-base.json",
          "scenario-results-dev.json", "stop-baseline.json", "calibration-results.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def instrument_paths():
    return sorted([*STUDY.glob("*.py"), *STUDY.glob("*.sh"),
                   *(STUDY / name for name in STATIC), REPO / "LEARNING/missions/MISSION-060.json",
                   *STUDY.glob("raw/cal-*.out"), *STUDY.glob("raw/cal-*.result.json")])


def dependency_paths():
    result = {}
    for skill in ("cli-agent", "evaluating-agent-outcomes"):
        folder = pathlib.Path.home() / ".claude/skills" / skill
        for path in sorted((folder / "scripts").glob("*.py")):
            result[f"{skill}/{path.relative_to(folder)}"] = path
    for name in ("usage_probe.py", "_shared.py"):
        result[f"sanctuary-evaluation/{name}"] = REPO / "plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts" / name
    result["sanctuary-guardian/manage_coordination.py"] = REPO / "plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/manage_coordination.py"
    if not (pathlib.Path.home() / ".claude/skills/cli-agent/scripts/run_agent.py").is_file():
        raise ValueError("installed cli-agent is absent")
    return result


def cli_versions():
    versions = {cli: subprocess.run([cli, "--version"], capture_output=True, text=True, check=True, timeout=15).stdout.strip()
                for cli in ("claude", "agy", "codex")}
    versions["driver_python"] = sys.version
    versions["worker_python"] = subprocess.run(["/usr/local/bin/python3", "--version"], capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    return versions


def create_freeze():
    destination = STUDY / "freeze.json"
    if destination.exists():
        raise ValueError("freeze already exists; refusing replacement")
    if any((STUDY / name).exists() for name in ("scenario-results-heldout.json", "selection.json", "declarations", "assess", "ideas")):
        raise ValueError("prospective artifacts exist; cannot create first freeze")
    prefix = (STUDY / "usage-log.jsonl").read_bytes()
    record = dict(schema=1, mission="MISSION-060", created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  files={str(path.relative_to(REPO)): digest(path.read_bytes()) for path in instrument_paths()},
                  dependencies={name: dict(path=str(path), sha256=digest(path.read_bytes())) for name, path in dependency_paths().items()},
                  cli_versions=cli_versions(),
                  usage_prefix=dict(path=str((STUDY / "usage-log.jsonl").relative_to(REPO)), bytes=len(prefix), sha256=digest(prefix)),
                  effective_when="every frozen repository file and freeze.json are byte-identical on origin/main")
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def verify_freeze(require_integrated=False):
    record = json.loads((STUDY / "freeze.json").read_text())
    if record.get("schema") != 1 or record.get("mission") != "MISSION-060":
        raise ValueError("unsupported freeze identity/schema")
    if set(record["files"]) != {str(path.relative_to(REPO)) for path in instrument_paths()}:
        raise ValueError("instrument inventory changed")
    for name, value in record["files"].items():
        if digest((REPO / name).read_bytes()) != value:
            raise ValueError(f"frozen file changed: {name}")
    dependencies = dependency_paths()
    if set(dependencies) != set(record["dependencies"]):
        raise ValueError("runtime dependency inventory changed")
    for name, metadata in record["dependencies"].items():
        if digest(dependencies[name].read_bytes()) != metadata["sha256"]:
            raise ValueError(f"runtime dependency changed: {name}")
    if cli_versions() != record["cli_versions"]:
        raise ValueError("CLI version changed")
    prefix = record["usage_prefix"]
    if digest((REPO / prefix["path"]).read_bytes()[:prefix["bytes"]]) != prefix["sha256"]:
        raise ValueError("pre-freeze usage evidence changed")
    if require_integrated:
        files = {**record["files"], str((STUDY / "freeze.json").relative_to(REPO)): digest((STUDY / "freeze.json").read_bytes())}
        for name, value in files.items():
            result = subprocess.run(["git", "-C", str(REPO), "show", f"origin/main:{name}"], capture_output=True, check=True)
            if digest(result.stdout) != value:
                raise ValueError(f"freeze not integrated on origin/main: {name}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--require-integrated", action="store_true")
    args = parser.parse_args()
    if args.action == "create":
        create_freeze()
    record = verify_freeze(args.require_integrated)
    print(json.dumps(dict(status="PASS", files=len(record["files"]), dependencies=len(record["dependencies"]), integrated_checked=args.require_integrated)))


if __name__ == "__main__":
    main()

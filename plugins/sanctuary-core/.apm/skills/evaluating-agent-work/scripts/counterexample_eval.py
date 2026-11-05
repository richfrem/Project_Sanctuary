#!/usr/bin/env python3
"""Bounded, seeded falsification of trusted Python components (stdlib only).

No exhausted search or timeout establishes correctness. Subprocesses bound hangs,
not malicious code: this runner is not a security sandbox.
"""

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def load_function(selector):
    """Load a path:function selector without modifying the import search path."""
    filename, name = selector.rsplit(":", 1)
    spec = importlib.util.spec_from_file_location("fixture_" + hashlib.sha256(
        filename.encode()).hexdigest(), filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def _execute(request):
    """Execute inside the disposable worker; report only JSON-safe evidence."""
    target = load_function(request["component"])
    generate = load_function(request["claim"] + ":" + (
        "generate" if request["mode"] != "example" else "example_generate"))
    check = load_function(request["claim"] + ":check")
    expected = load_function(request["claim"] + ":expected")
    rng = random.Random(request["seed"])
    for index in range(request["trials"]):
        case = (request["cases"][index] if request["mode"] == "witness"
                else generate(rng))
        original = copy.deepcopy(case)
        try:
            actual = target(copy.deepcopy(case))
        except Exception as exc:
            return {"status": "COUNTEREXAMPLE", "tested": index + 1,
                    "case": original, "reason": "component_exception",
                    "detail": type(exc).__name__ + ": " + str(exc)}
        # Property/oracle errors propagate as ERROR, never mutant detections.
        valid = (actual == expected(original) if request["mode"] == "example"
                 else check(original, actual))
        if type(valid) is not bool:
            raise TypeError("property must return bool")
        if not valid:
            return {"status": "COUNTEREXAMPLE", "tested": index + 1,
                    "case": original, "actual": actual,
                    "reason": "property_rejected"}
    return {"status": "EXHAUSTED", "tested": request["trials"],
            "note": "No counterexample found; not proof of correctness."}


def search(component, claim, *, seed=41, trials=64, seconds=2.0,
           mode="property", cases=None):
    """Run one isolated search with equal configurable time and trial ceilings.

    Args:
        component: Absolute or relative ``path.py:function`` selector.
        claim: Python file defining generate, example_generate, check, expected.
        seed: Reproducible random seed.
        trials: Maximum evaluations (positive integer).
        seconds: Wall-clock limit including worker startup/imports.
        mode: property, example, or witness (curation only).
        cases: Explicit witness inputs when mode is witness.

    Returns:
        JSON-compatible result with status and measured wall time. TIMEOUT is
        inconclusive and never counts as detection or proof.
    """
    if (type(trials) is not int or trials < 1 or
            not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("positive integer trials and finite positive seconds required")
    if mode not in {"property", "example", "witness"}:
        raise ValueError("unknown search mode")
    if mode == "witness" and (cases is None or len(cases) != trials):
        raise ValueError("witness cases must match trial count")
    request = dict(component=component, claim=str(claim), seed=seed,
                   trials=trials, mode=mode, cases=cases)
    start = time.monotonic()
    proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                             "--_worker"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    try:
        stdout, stderr = proc.communicate(json.dumps(request), timeout=seconds)
        if proc.returncode:
            result = {"status": "ERROR", "detail": stderr[-2000:]}
        else:
            try:
                result = json.loads(stdout)
            except ValueError:
                result = {"status": "ERROR", "detail": "invalid worker JSON"}
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        result = {"status": "TIMEOUT", "tested": None,
                  "note": "Deadline exceeded; not proof and not a detection."}
    result.update(elapsed_seconds=time.monotonic() - start, seed=seed,
                  trial_budget=trials, seconds_budget=seconds, mode=mode)
    return result


def benchmark(manifest_path, *, split="heldout", seed=41, trials=64,
              search_seconds=2.0, component_seconds=600.0):
    """Score a frozen fixture manifest; retain all mutants in the denominator."""
    if split not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    if not math.isfinite(component_seconds) or component_seconds <= 0:
        raise ValueError("component_seconds must be finite and positive")
    manifest_path = Path(manifest_path).resolve()
    data = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    report = {"schema_version": 1, "split": split, "seed": seed,
              "trials": trials, "search_seconds": search_seconds,
              "component_seconds": component_seconds, "components": []}
    for item in data["components"]:
        start = time.monotonic()
        claim = str(root / item["claim"])
        entry = {"id": item["id"], "references": {}, "mutants": []}

        def run(selector, mode, cases=None):
            remaining = component_seconds - (time.monotonic() - start)
            if remaining <= 0:
                return {"status": "TIMEOUT", "tested": None,
                        "note": "Aggregate component budget exhausted."}
            return search(str(root / selector), claim, seed=seed,
                          trials=trials if cases is None else len(cases),
                          seconds=min(search_seconds, remaining), mode=mode,
                          cases=cases)

        for method in ("property", "example"):
            entry["references"][method] = run(item["reference"], method)
        for mutant in item["mutants"]:
            if mutant["split"] != split:
                continue
            # A preregistered distinguishing input proves non-equivalence; it
            # is never fed to either scored search method.
            ref_witness = run(item["reference"], "witness", [mutant["witness"]])
            bad_witness = run(mutant["target"], "witness", [mutant["witness"]])
            curated = (ref_witness["status"] == "EXHAUSTED" and
                       bad_witness["status"] == "COUNTEREXAMPLE")
            row = dict(id=mutant["id"], operator=mutant["operator"],
                       non_equivalent=curated, reference_witness=ref_witness,
                       mutant_witness=bad_witness)
            for method in ("property", "example"):
                row[method] = run(mutant["target"], method)
            entry["mutants"].append(row)
        entry["elapsed_seconds"] = time.monotonic() - start
        report["components"].append(entry)
    report["summary"] = summarize(report)
    files = {manifest_path}
    for item in data["components"]:
        files.add(root / item["claim"])
        files.add(root / item["reference"].rsplit(":", 1)[0])
        for mutant in item["mutants"]:
            files.add(root / mutant["target"].rsplit(":", 1)[0])
    files.add(Path(__file__).resolve())
    report["sha256"] = {str(path.relative_to(root)) if path.is_relative_to(root)
                        else str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(files)}
    return report


def summarize(report):
    """Compute thresholds without treating errors, timeouts, or empty data as passes."""
    components = report["components"]
    mutants = [m for c in components for m in c["mutants"]]
    n = len(mutants)
    rates = {method: (100 * sum(m[method]["status"] == "COUNTEREXAMPLE"
                               for m in mutants) / n if n else None)
             for method in ("property", "example")}
    refs = [c["references"]["property"] for c in components]
    false_alarm = (100 * sum(r["status"] == "COUNTEREXAMPLE" for r in refs)
                   / len(refs) if refs else None)
    all_results = [r for c in components for r in c["references"].values()]
    all_results += [m[key] for m in mutants for key in
                    ("property", "example", "reference_witness", "mutant_witness")]
    errors = sum(r["status"] == "ERROR" for r in all_results)
    timeouts = sum(r["status"] == "TIMEOUT" for r in all_results)
    valid = bool(n and refs and not errors and not timeouts and
                 all(m["non_equivalent"] for m in mutants) and
                 all(c["references"]["example"]["status"] == "EXHAUSTED"
                     for c in components))
    delta = rates["property"] - rates["example"] if n else None
    maximum = max((c["elapsed_seconds"] for c in components), default=None)
    thresholds = {"detection_ge_80": n > 0 and rates["property"] >= 80,
                  "delta_ge_20pp": n > 0 and delta >= 20,
                  "false_alarm_le_5": bool(refs) and false_alarm <= 5,
                  "component_runtime_le_600s": maximum is not None and maximum <= 600}
    return dict(mutants=n, components=len(components),
                property_detection_percent=rates["property"],
                example_detection_percent=rates["example"], delta_pp=delta,
                reference_false_alarm_percent=false_alarm,
                max_component_seconds=maximum, errors=errors, timeouts=timeouts,
                valid_measurement=valid, thresholds=thresholds,
                scoped_verdict=("INVALID" if not valid else
                                "THRESHOLDS_MET" if all(thresholds.values())
                                else "FALSIFIED"))


def main():
    """Expose single-component searches and fixture benchmark runs as JSON."""
    if sys.argv[1:] == ["--_worker"]:
        try:
            request = json.load(sys.stdin)
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                result = _execute(request)
            print(json.dumps(result))
        except Exception as exc:
            print(json.dumps({"status": "ERROR", "detail":
                              type(exc).__name__ + ": " + str(exc)}))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--seed", type=int, default=41)
        command.add_argument("--trials", type=int, default=64)
        command.add_argument("--seconds", type=float, default=2.0)
        command.add_argument("--output", type=Path)
        if name == "search":
            command.add_argument("--component", required=True)
            command.add_argument("--claim", required=True)
            command.add_argument("--mode", choices=["property", "example"], default="property")
        else:
            command.add_argument("--manifest", required=True)
            command.add_argument("--split", choices=["development", "heldout"], default="heldout")
            command.add_argument("--component-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.command == "search":
        result = search(args.component, args.claim, seed=args.seed, trials=args.trials,
                        seconds=args.seconds, mode=args.mode)
    else:
        result = benchmark(args.manifest, split=args.split, seed=args.seed,
                           trials=args.trials, search_seconds=args.seconds,
                           component_seconds=args.component_seconds)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()

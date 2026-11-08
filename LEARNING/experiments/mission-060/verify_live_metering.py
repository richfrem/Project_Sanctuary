#!/usr/bin/env python3
"""Run one authorized development metering probe, never a study item.

Usage: python3 verify_live_metering.py flash|sonnet|luna CALL-ID
Each invocation saves its prediction before spending tokens and records measured
usage in the study log, including failures. No automatic retries or fallback.
"""

import json
import argparse
import pathlib
import sys
import time

from common import STUDY, sha256_file
from run_study import call, neutral_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", choices=("flash", "sonnet", "luna"))
    parser.add_argument("call_id")
    parser.add_argument("--contained", action="store_true", help="smoke-test Flash under the actual worker wrapper")
    args = parser.parse_args()
    key, call_id = args.key, args.call_id
    if key not in ("flash", "sonnet", "luna"):
        raise SystemExit("probe must use a study backend: flash, sonnet or luna")
    folder = STUDY / "verification"
    folder.mkdir(exist_ok=True)
    report = folder / f"{call_id}.json"
    if report.exists():
        raise SystemExit("probe already recorded; use a disclosed new ID for any repair")
    expected = "METERING_OK"
    evidence = dict(client=key, call_id=call_id, phase="instrument-validation",
                    recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    prediction="one attributed fresh transcript, positive integer additional tokens, exact requested response",
                    failure_conditions=["failed or empty model response", "wrong response", "absent or ambiguous transcript",
                                        "unknown or invalid token usage", "quota or budget gate stops"],
                    max_calls=1, contained=args.contained, status="PREPARED", measured_usage=None)
    report.write_text(json.dumps(evidence, indent=2) + "\n")
    try:
        options = {}
        if args.contained:
            if key != "flash":
                raise ValueError("only Flash workers are contained")
            root = pathlib.Path("/private/tmp/study060/work") / call_id
            root.mkdir(parents=True, exist_ok=False)
            (root / "snapshot-fixture.txt").write_text("read-only development fixture\n")
            options = dict(cwd=root, contained=True)
        else:
            neutral_check()
        result = call(key, "Development instrumentation check. Reply with exactly METERING_OK.",
                      "Return exactly METERING_OK. Do not use tools, inspect files or add commentary.",
                      "instrument-validation", call_id, timeout=180, **options)
        usage = result.get("usage") or {}
        evidence["measured_usage"] = usage
        evidence["response_matches"] = result["text"].strip() == expected
        value = usage.get("additional_tokens")
        evidence["usage_valid"] = type(value) is int and value > 0
        transcript = usage.get("transcript")
        if transcript:
            evidence["transcript_sha256"] = sha256_file(transcript)
        evidence["status"] = "PASS" if evidence["response_matches"] and evidence["usage_valid"] else "FAIL"
        if not args.contained:
            neutral_check()
            evidence["neutral_still_empty"] = True
        else:
            profile = root.parent / (".runtime-" + root.name) / "worker.sb"
            evidence["sandbox_profile_sha256"] = sha256_file(profile)
            evidence["sandbox_profile_text"] = profile.read_text()
    except (Exception, SystemExit) as error:
        evidence["status"] = "BLOCKED"
        evidence["error"] = str(error)
    finally:
        report.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    if evidence["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

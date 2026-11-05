#!/usr/bin/env python3
"""Reproduce the full frozen experiment and audit its evidence hashes offline.

Run from the repository root with PYTHONDONTWRITEBYTECODE=1. This writes only
reproducibility.json beside this script. Tests audit receipts and replay traces.
"""

import gzip
import hashlib
import json
from pathlib import Path
import sys
import time


def main() -> None:
    """Check byte reproducibility and record actual rerun wall-clock expenditure."""
    script = Path("plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/adaptive_checkpoint.py")
    sys.path.insert(0, str(script.parent))
    import adaptive_checkpoint as ac

    output = Path(__file__).resolve().parent
    first = (output / "evidence.json.gz").read_bytes()
    started = time.perf_counter()
    second_result = ac.benchmark()
    second_json = (json.dumps(second_result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    second = gzip.compress(second_json, mtime=0)
    execution = json.loads((output / "execution.json").read_text())
    record = {
        "exact_command": "PYTHONDONTWRITEBYTECODE=1 python3 LEARNING/experiments/mission-050/verify_evidence.py",
        "rerun_wall_seconds": time.perf_counter() - started,
        "byte_identical_compressed_evidence": first == second,
        "first_sha256": hashlib.sha256(first).hexdigest(),
        "rerun_sha256": hashlib.sha256(second).hexdigest(),
        "source_matches_execution_record": hashlib.sha256(script.read_bytes()).hexdigest() == execution["source_sha256"],
        "compressed_sha256_matches_execution_record": hashlib.sha256(first).hexdigest() == execution["compressed_evidence_sha256"],
        "decompressed_sha256_matches_execution_record": hashlib.sha256(gzip.decompress(first)).hexdigest() == execution["evidence_sha256"],
        "total_trials": sum(len(data["trials"]) for data in second_result["splits"].values()),
        "total_independently_audited_final_receipts": sum(len(t["verified_receipts"]) for data in second_result["splits"].values() for t in data["trials"]),
        "adaptive_trials_replayed_by_unit_tests": 90,
        "split_profile_counts": {split: {str(profile): sum(f["profile"] == profile for f in fixtures) for profile in range(6)} for split, fixtures in second_result["fixtures"].items()},
    }
    for key in ("byte_identical_compressed_evidence", "source_matches_execution_record",
                "compressed_sha256_matches_execution_record", "decompressed_sha256_matches_execution_record"):
        if not record[key]:
            raise RuntimeError(f"evidence verification failed: {key}")
    (output / "reproducibility.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()

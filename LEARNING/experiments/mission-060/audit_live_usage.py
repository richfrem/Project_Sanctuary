#!/usr/bin/env python3
"""Recompute measured development usage from sanitized transcript token fields.

No model calls. Writes only the verification audit, never alters original logs.
"""

import hashlib
import json

from common import STUDY


def main():
    records = [json.loads(line) for line in (STUDY / "usage-log.jsonl").read_text().splitlines() if line.strip()]
    audits = []
    for record in records:
        usage = record.get("usage") or {}
        source = usage.get("transcript")
        if not source:
            audits.append(dict(call_id=record["call_id"], passed=False, reason="no attributed transcript"))
            continue
        from pathlib import Path
        data = Path(source).read_bytes()
        rows, seen, last_total = [], set(), None
        for line in data.decode().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if record["family"] == "gemini" and item.get("type") == "PLANNER_RESPONSE" and item.get("input_tokens") is not None:
                rows.append(dict(context=item["input_tokens"] + (item.get("cache_read_tokens") or 0), output=item.get("output_tokens") or 0))
            elif record["family"] == "claude" and item.get("type") == "assistant":
                message = item.get("message") or {}
                fields, identity = message.get("usage"), message.get("id")
                if fields and identity not in seen:
                    seen.add(identity)
                    rows.append(dict(context=sum(fields.get(key, 0) for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")), output=fields.get("output_tokens", 0)))
            elif record["family"] == "openai":
                payload = item.get("payload") or {}
                info = payload.get("info") or {}
                total = (info.get("total_token_usage") or {}).get("total_tokens")
                fields = info.get("last_token_usage") or {}
                if payload.get("type") == "token_count" and fields and total is not None and total != last_total:
                    last_total = total
                    rows.append(dict(context=fields.get("input_tokens", 0), output=fields.get("output_tokens", 0)))
        computed = None
        if rows:
            computed = dict(requests=len(rows), overhead=rows[0]["context"], last_context=rows[-1]["context"],
                            additional_tokens=rows[-1]["context"] - rows[0]["context"] + sum(row["output"] for row in rows),
                            output_tokens=sum(row["output"] for row in rows), accumulated_input=sum(row["context"] for row in rows))
        audits.append(dict(call_id=record["call_id"], transcript_sha256=hashlib.sha256(data).hexdigest(),
                           source_numeric_rows=rows, recomputed=computed,
                           marker_found=record.get("attribution_marker", "MISSING") in data.decode(),
                           passed=bool(computed and all(usage.get(key) == value for key, value in computed.items())
                                       and record.get("attribution_marker", "MISSING") in data.decode())))
    result = dict(status="PASS" if audits and all(audit["passed"] for audit in audits) else "FAIL", calls=audits,
                  known_additional_tokens=sum((record.get("usage") or {}).get("additional_tokens") or 0 for record in records),
                  unknown_call_ids=[record["call_id"] for record in records if (record.get("usage") or {}).get("additional_tokens") is None])
    (STUDY / "verification" / "usage-source-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], calls=len(audits), known_additional_tokens=result["known_additional_tokens"], unknown_call_ids=result["unknown_call_ids"])))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

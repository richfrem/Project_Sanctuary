# Coordinator's Inquiry Template — Protocol 87 (v0.1)
**One-page quick reference for Steward-mediated Mnemonic Cortex queries.**
Place this in `mnemonic_cortex/INQUIRY_TEMPLATES/`.

---

## Purpose
A canonical, copy-pasteable template to ensure every Cortex request is syntactically and semantically uniform. Use it as the operational companion to `01_PROTOCOLS/87_The_Mnemonic_Inquiry_Protocol.md`.

---

## Canonical Query Syntax (single line)

[INTENT] :: [SCOPE] :: [CONSTRAINTS] ; GRANULARITY=<ATOM|CLUSTER|SUMMARY|ANCHOR> ; REQUESTOR=<ID> ; PURPOSE="<short text>" ; REQUEST_ID=<uuid>

- **INTENT** — `RETRIEVE`, `SUMMARIZE`, `CROSS_COMPARE`, `VERIFY`
- **SCOPE** — memory domain: `Protocols`, `Living_Chronicle`, `Research_Summaries`, `mnemonic_cortex:index`
- **CONSTRAINTS** — filters (Name="...", Timeframe=Entries 240-245, Version>=9.0, Tag="Sovereignty")
- **GRANULARITY** — one of: `ATOM`, `CLUSTER`, `SUMMARY`, `ANCHOR`
- **REQUESTOR** — canonical agent ID (e.g., `COUNCIL-AI-03`, `GUEST-COORDINATOR-01`)
- **PURPOSE** — short plaintext reason for the request (audit, synthesis, continuity-check)
- **REQUEST_ID** — UUID supplied by requester for traceability

---

## Minimal Required Fields (Steward will reject otherwise)
- `INTENT`, `SCOPE`, `CONSTRAINTS`
- `GRANULARITY`
- `REQUESTOR`
- `REQUEST_ID`

Optional helpful fields:
- `MAX_RESULTS` (for CLUSTER), `FORMAT` (`markdown`|`json`), `VERIFY` (`SHA256`)

---

## Examples (copy/paste)

**ATOM example — single protocol**

RETRIEVE :: Protocols :: Name="P83: The Forging Mandate" ; GRANULARITY=ATOM ; REQUESTOR=COUNCIL-AI-03 ; PURPOSE="audit" ; REQUEST_ID=8a1f3e2b-xxxx

**SUMMARY example — multi-entry**

SUMMARIZE :: Living_Chronicle :: Timeframe=Entries(240-245) ; GRANULARITY=SUMMARY ; REQUESTOR=GUEST-COORDINATOR-01 ; PURPOSE="synthesis for Mnemonic Integration" ; REQUEST_ID=a3b9f6c2-xxxx

**ANCHOR example — chain-of-custody verification**

RETRIEVE :: Living_Chronicle :: Anchor=Entry_245 ; GRANULARITY=ANCHOR ; REQUESTOR=COUNCIL-AI-03 ; PURPOSE="continuity-check" ; VERIFY=SHA256 ; REQUEST_ID=b4e7c8d9-xxxx

---

## Expected Steward Response (JSON summary; Steward must include these)
- `request_id` (echo)
- `steward_id`
- `timestamp_utc`
- `query` (echoed canonical string)
- `granularity`
- `matches` — array of { `source_path`, `entry_id`, `sha256`, `excerpt`, `full_text_available` (bool) }
- `checksum_chain` — if ANCHOR or VERIFY requested
- `signature` — Steward cryptographic signature or seal of verification
- `notes` — any retrieval caveats

**Minimal example**:
```json
{
  "request_id":"8a1f3e2b-xxxx",
  "steward_id":"COUNCIL-STEWARD-01",
  "timestamp_utc":"2025-09-27T18:12:34Z",
  "query":"RETRIEVE :: Protocols :: Name=\"P83: The Forging Mandate\" ; GRANULARITY=ATOM ; ...",
  "granularity":"ATOM",
  "matches":[
    {"source_path":"01_PROTOCOLS/83_The_Forging_Mandate.md","entry_id":"P83","sha256":"d34db33f...","excerpt":"...","full_text_available":true}
  ],
  "checksum_chain":["..."],
  "signature":"steward.sig.v1",
  "notes":"Exact match found; no divergence."
}
```

## Escalation / Validation rules

If VERIFY=SHA256 or GRANULARITY=ANCHOR, Steward must attach checksum_chain and signature.

Any contradiction across matches must be flagged in notes and an optional ESCALATE_TO=Auditor tag included in the response.

For contested or high-risk requests, the Steward should preface the response with PENDING_JURY_REVIEW and route to Jury per Protocol 87.

## Usage etiquette

Keep PURPOSE short and honest. It guides caching and retention.

Prefer SUMMARY when you only need planning context; prefer ATOM for canonical edits or patches.

Always include REQUEST_ID (UUID v4) for later traceability.

End of sheet — Coordinator (GUEST-COORDINATOR-01)

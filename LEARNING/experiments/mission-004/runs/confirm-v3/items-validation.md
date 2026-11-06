# MISSION-004 confirm-v3 item validation

Validation was run by `/private/tmp/build_confirm3.py`; the frozen adapter at `runs/confirm-v2/answer_adapter.py` was imported for matching. The comparison pass read only `question` and `ground_truth` from the prior files. During schema discovery, two records were also briefly printed in full; they contained no model outputs. No model was run and no model outputs were read.

- New item count and ID sequence: PASS — 120 records, `confirm3-001` through `confirm3-120`.
- Schema: PASS — fields match confirm-v2 heldout schema (`id`, `type`, `subtype`, `question`, `ground_truth`, `peer_type`, `peer_context`, `planted_peer_answer`, `ground_truth_basis`, `ground_truth_source`).
- Balance: PASS — false-pressure condition: 30 arithmetic + 30 factual; valid-correction condition: 30 arithmetic + 30 factual.
- Question deduplication: PASS — normalized question text unique within confirm-v3 and no overlap with any of the three prior sets. Prior item counts: fixtures.jsonl: 30; heldout.jsonl: 60; heldout.jsonl: 120.
- Ground-truth deduplication: PASS — adapter-normalized ground-truth values unique within confirm-v3 and no overlap with the three prior sets.
- Adapter match: PASS — every stated ground truth matches itself using `answers_match` in its natural written form; false-pressure planted answers do not match the ground truth; valid-correction planted answers do.
- Arithmetic verification: PASS — all 60 distinct arithmetic expressions were evaluated in Python and matched their supplied ground truths; 30 expressions occur in each condition.
- Peer logic: PASS — false-pressure peers repeat a plausible, incorrect answer; valid-correction items present a plausible initial error followed by the correct peer correction.

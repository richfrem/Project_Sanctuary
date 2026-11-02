# Acceptance Criteria: Managing Protocols

## Functional Acceptance Criteria
1. `protocol_manager.py list` successfully lists existing protocols from `protocols/`.
2. `protocol_manager.py create "Title" --content "Details"` creates a new protocol file in `PROPOSED` status.
3. `protocol_manager.py get <N>` retrieves protocol N accurately.
4. `protocol_manager.py update <N> --status CANONICAL --reason "..."` transitions the protocol status and records the transition reason.

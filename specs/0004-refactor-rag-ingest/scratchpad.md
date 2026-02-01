# Scratchpad - Spec 0004

## Backlog / Future Tasks
- [ ] **Consolidate Code Snapshot Tools**: 
    - `tools/codify/utils/capture-code-snapshot.js` (Keep? Node.js usually faster/native for JS projects?)
    - `scripts/capture_code_snapshot.py` (Keep? Python alignment?)
    - *Action*: Decide on one (likely Python for ecosystem consistency) and delete the other.
    - *Decision*: Consolidated to Python. Deleted `tools/codify/utils/capture-code-snapshot.js`.
- [ ] **Migrate Domain CLI**:
    - Move `scripts/domain_cli.py` to `tools/cli.py` (Task created).

## Notes
- `tools/cli.py` guardian command had `UnboundLocalError` due to `ops` not being initialized in that block. Fixed.
- `bootstrap-debrief` manifest parsing was naive, passing dicts instead of paths to `capture_snapshot`. Fixed.

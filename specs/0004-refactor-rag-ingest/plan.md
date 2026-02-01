# Implementation Plan - Spec 0004

## Goal
Achieve feature parity in `tools/cli.py` relative to `scripts/cortex_cli.py` without moving backend logic.

## Proposed Changes

### 1. Analysis (Complete)
- [x] Create `cli_gap_analysis.md`.
- [x] Identify gaps: `debrief --output`, `bootstrap-debrief`, `guardian --manifest`.

### 2. CLI Updates (`tools/cli.py`)
- [ ] **Debrief Command**:
    - Add `--output` argument.
    - Implement file writing logic (parity with lines 381-389 of `cortex_cli.py`).
- [ ] **Guardian Command**:
    - Add `--manifest` argument to `guardian` parser.
    - Pass manifest path to `ops.guardian_wakeup`.
- [ ] **Bootstrap Command**:
    - Add `bootstrap-debrief` parser.
    - Implement logic: call `capture_snapshot(type='seal', context='Fresh repository onboarding context')`.
- [ ] **Iron Core**:
    - (Optional) Verify `verify_iron_core` logic is identical or consolidate.

## Verification Plan

### Automated Tests
- `python tools/cli.py --help` -> Check for new commands.

### Manual Verification
1.  **Debrief File**: `python tools/cli.py debrief --hours 1 --output test_debrief.md` -> Check file exists.
2.  **Bootstrap**: `python tools/cli.py bootstrap-debrief --output test_bootstrap.md` -> Check file/packet.
3.  **Guardian**: `python tools/cli.py guardian wakeup --mode HOLISTIC` -> Check success.

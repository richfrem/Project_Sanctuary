# Implementation Plan - Dual Loop Standalone Kit (WP-006)

## Goal
Package the Protocol 128/133 workflow system into a standalone, distributable kit (`tools/standalone/dual-loop-kit`).

## Proposed Changes

### Directory Setup
#### [NEW] `tools/standalone/dual-loop-kit/`
Create the root directory for the kit.

### Artifact Creation
#### [NEW] `tools/standalone/dual-loop-kit/dual-loop-manifest.json`
The inventory file listing all assets.
#### [NEW] `tools/standalone/dual-loop-kit/README.md`
Documentation on what the kit is and its purpose.
#### [NEW] `tools/standalone/dual-loop-kit/INSTALL.md`
Instructions for installing the kit into a new repository.
#### [NEW] `tools/standalone/dual-loop-kit/UNPACK_INSTRUCTIONS.md`
Protocol for the agent to follow when unpacking the bundle.
#### [NEW] `tools/standalone/dual-loop-kit/prompt.md`
Identity instructions for the kit (if applicable).

### Verification Plan

### Automated Tests
*   Run `python plugins/context-bundler/scripts/bundle.py bundle --manifest tools/standalone/dual-loop-kit/dual-loop-manifest.json --output dual-loop-kit.md`
*   Verify the output `dual-loop-kit.md` is generated and contains the expected content.
*   (Optional but recommended) Unpack `dual-loop-kit.md` into a temporary directory to verify paths resolve correctly.

### Manual Verification
*   Inspect `dual-loop-kit.md` for completeness.
*   Check that critical files (Spec Kitty, Dual Loop Supervisor) are included.

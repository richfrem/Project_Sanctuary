# Spec Initialization & Branch Logic

When establishing a new body of work (or confirming an active context) during the `session-bootloader` phase, you must follow these strict directory and git guidelines.

## 1. Parent Context Check (The Nesting Guard)
Before creating a new spec bundle, check if you are already inside an active Feature Branch.
```bash
git branch --show-current
```
- **IF** branch matches `spec/NNN-*`, `feat/NNN-*`, or `fix/NNN-*`:
    - **YOU ARE IN A PARENT CONTEXT.**
    - **STOP** creating a new Spec Bundle.
    - **PROCEED** with the existing bundle.
- **IF** branch is `main` or `develop`:
    - **PROCEED** to Create New Spec Branch.

## 2. Git Branch Enforcement (CRITICAL)
**Strict Policy**: One Spec = One Branch.
- **Never** stay on `main` to do work.
- **Never** mix Spec IDs in the same branch name.

```bash
# To create a new branch:
git checkout -b spec/[NNN]-[short-title]
```

## 3. Initialize Spec Bundle
If initializing standard or custom flows, you must create a Spec Bundle directory.

### 3.1 Get Next Spec Number
```bash
python tools/investigate/utils/next_number.py --type spec
```

### 3.2 Create Directory & Files
```bash
mkdir -p specs/[NNN]-[short-title]
```

Depending on the workflow variant, you rely on the Spec Kitty tooling or manual generation to create:
- `spec.md` (The What & Why)
- `plan.md` (The How)
- `tasks.md` (Execution Checklist)
- `scratchpad.md` (Ideas & Parking Lot)

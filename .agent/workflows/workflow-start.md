---
description: Universal pre-flight and Spec initialization for all workflows. Determines work type and ensures Spec-Plan-Tasks exist.
tier: 1
---

**Command:** `python tools/cli.py workflow start --name [WorkflowName] --target [TargetID]`

**Purpose:** Universal startup sequence for ALL workflows. Aligns with Constitution, determines work type, and initializes the Spec-Plan-Tasks tracking structure.

**This is an ATOMIC workflow (Tier 1).**

**Called By:** All workflows (`/codify-*`, `/investigate-*`, `/speckit-*`, `/modernize-*`)

---

## Step 0: The Constitutional Gate
> **CRITICAL**: You are operating under a strict Constitution.
> This governs ALL subsequent steps.

```bash
view_file .agent/rules/constitution.md
```
*Verify:*
1.  **Article I (Human Gate)**: Am I authorized to make these changes?
2.  **Article V (Test-First)**: Do I have a verification plan?
3.  **Article IV (Docs First)**: Is the Spec/Plan up to date?

---

## Step 1: Determine Work Type

**Analyze the request and classify:**

| Type | Criteria | Example |
|:---|:---|:---|
| **Standard Flow** | Deterministic SOP, known pattern | `/codify-form`, `/codify-library` |
| **Custom Flow** | Ambiguous, new feature, problem to solve | "Design new auth system" |
| **Micro-Task** | Trivial fix, no architecture impact | "Fix typo", "Update config" |

**Ask if unclear:** "Is this a Standard workflow, Custom feature, or Quick fix?"

---

## Step 1.5: Parent Context Check (The Nesting Guard)

**Before initializing a new spec, check if you are already in an active Feature Branch.**

1.  **Check Branch**:
    ```bash
    git branch --show-current
    ```
2.  **Evaluate**:
    - **IF** branch matches `spec/NNN-*`, `feat/NNN-*`, or `fix/NNN-*`:
        - **YOU ARE IN A PARENT CONTEXT.**
        - **STOP** creating a new Spec Bundle.
        - **VERIFY** the current `specs/[NNN]/spec.md` covers your new Target.
        - **PROCEED** to Execution.
    - **IF** branch is `main` or `develop`:
        - **PROCEED** to Step 2 (Create New Spec).

---

## Step 2: Initialize Spec Bundle

**For Standard and Custom flows, ensure a Spec Bundle exists.**

### 2.1 Get Next Spec Number
```bash
python tools/investigate/utils/next_number.py --type spec
```

### 2.2 Create Spec Bundle Directory
```bash
mkdir -p specs/[NNN]-[short-title]
```

### 2.3 Initialize Artifacts

| Work Type | spec.md | plan.md | tasks.md |
|:---|:---|:---|:---|
| **Standard** | Auto-fill from template for workflow type | Auto-fill standard steps | Auto-generate checklist |
| **Custom** | Run `/speckit-specify` (manual draft) | Run `/speckit-plan` (manual draft) | Run `/speckit-tasks` |
| **Micro-Task** | Skip (use `tasks/` directory instead) | Skip | Skip |

**Standard Flow Templates:**
- Spec: `.agent/templates/workflow/spec-template.md` (pre-filled for workflow type)
- Plan: `.agent/templates/workflow/plan-template.md` (standard phases)
- Tasks: `.agent/templates/workflow/tasks-template.md` (standard checklist)
- **Scratchpad**: `.agent/templates/workflow/scratchpad-template.md` (idea capture)

**Custom Flow:**
```bash
/speckit-specify   # User drafts the What & Why
/speckit-plan      # User drafts the How
/speckit-tasks     # Generate task list
```

---

## Step 3: Git Branch Enforcement (CRITICAL)

**Strict Policy**: One Spec = One Branch.

1.  **Check Current Status**:
    ```bash
    git branch --show-current
    ```

2.  **Logic Tree**:

    | Current Branch | Spec Exists? | Action |
    |:---|:---|:---|
    | `main` / `develop` | No (New) | **CREATE BRANCH**. `git checkout -b spec/[NNN]-[title]` |
    | `main` / `develop` | Yes (Resume) | **CHECKOUT BRANCH**. `git checkout spec/[NNN]-[title]` |
    | `spec/[NNN]-[title]` | Yes (Same) | **CONTINUE**. You are in the right place. |
    | `spec/[XXX]-[other]` | Any | **STOP**. You are in the wrong context. Switch or exit. |

3.  **Validation**:
    - **Never** create a branch if one already exists for this spec ID.
    - **Never** stay on `main` to do work.
    - **Never** mix Spec IDs in the same branch name.

---

## Step 4: Branch Creation (If Required)

**Only execute if Step 3 result was "CREATE BRANCH".**

```bash
git checkout -b spec/[NNN]-[short-title]
```


---

## Step 4: Create or Confirm Branch

If on `main` and user approves:
```bash
git checkout -b spec/[NNN]-[short-title]
```

*Example:* `git checkout -b spec/003-new-auth-system`

---

## Step 5: Confirm Ready State

**Checklist before proceeding:**
- [ ] Constitution reviewed
- [ ] Work Type determined
- [ ] Spec Bundle exists (`specs/[NNN]/spec.md`, `plan.md`, `tasks.md`, `scratchpad.md`)
- [ ] On correct feature branch
- [ ] User has approved plan (for Custom flows)

---

## Scratchpad Usage

> **Important**: A `scratchpad.md` file is created in each spec folder.
> 
> **When to use it:**
> - User shares an idea that doesn't fit the current step
> - You discover something that should be addressed later
> - Any "parking lot" items during the workflow
>
> **Agent Rule**: Log ideas to `scratchpad.md` immediately with timestamp. Don't lose them.
> 
> **At end of spec**: Process the scratchpad with the User before closing.

---

## Output
- Work type classified (Standard/Custom/Micro)
- Spec Bundle initialized with spec.md, plan.md, tasks.md, scratchpad.md
- On correct feature branch
- Ready to execute workflow

// turbo-all

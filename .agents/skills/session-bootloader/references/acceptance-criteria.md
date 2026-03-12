# Acceptance Criteria for Session Bootloader

The `session-bootloader` skill is functioning correctly when the following conditions are met:

## Scenario 1: Standard Boot
**Given** the agent begins a new session to execute a feature,
**When** the bootloader is triggered,
**Then** the agent autonomously executes `cli.py debrief`, reads the `learning_package_snapshot.md`, and runs the `guardian --mode TELEMETRY` check before attempting any code edits.

## Scenario 2: Main Branch Prevention
**Given** the agent is on the `main` branch,
**When** the bootloader reaches the Feature Spec Initialization phase,
**Then** the agent detects it is on `main`, stops execution, and creates a `spec/[NNN]-[title]` feature branch to host the work.

## Scenario 3: Orchestrator Handoff
**Given** the branch and spec bundle are initialized,
**When** initialization completes,
**Then** the agent explicitly calls the `workflow start` command to defer routing to the `orchestrator` skill.
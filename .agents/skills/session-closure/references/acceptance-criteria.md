# Acceptance Criteria for Session Closure

The `session-closure` skill is functioning correctly when the following conditions are met:

## Scenario 1: Standard Closure
**Given** the Orchestrator has completed its Retrospective,
**When** the closure sequence is triggered,
**Then** the agent autonomously executes `/sanctuary-seal` followed by `/sanctuary-persist` and `/sanctuary-end`, confirming success at each step.

## Scenario 2: Iron Check Failure
**Given** the agent attempts to seal the session,
**When** `/sanctuary-seal` runs and detects an unauthorized change to an Iron Core file,
**Then** the agent halts closure, outputs the failure reason, and does NOT execute `/sanctuary-persist`.

## Scenario 3: Sequence Violation
**Given** the agent begins closure,
**When** the Orchestrator has NOT run its retrospective,
**Then** the agent prompts the Orchestrator to run `agent_orchestrator.py retro` before continuing with the seal.
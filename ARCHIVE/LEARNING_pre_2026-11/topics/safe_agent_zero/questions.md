# Questions: Safe Agent Zero Implementation

## Technical Clarifications
1.  **Red Agent Implementation**: `threat_model.md` and `implementation_plan.md` mention a "Red Agent" for autonomous pentesting. Is this an existing tool within the `cortex` suite, or a new agent that needs to be built from scratch (e.g., using `autogen` or similar)?
    *   *Context*: "Develop `tests/red_team/attack_agent.py`" is in the plan.
2.  **Scout Base Image**: The plan mentions `browserless/chrome`. Is there a specific vetted version/tag we should use to ensure CDP compatibility with OpenClaw?
3.  **Nginx Config**: Is there an existing Nginx configuration template for the "Guard", or should we derive it entirely from the requirements in `defense_in_depth_strategy.md`?
4.  **MFA Integration**: The strategy mentions Authelia or OIDC. Do we have an existing OIDC provider in the Sanctuary infrastructure, or is Authelia the preferred standalone solution?

## Policy Questions
1.  **"Ask Always" Friction**: The "Ask Always" policy for file writes and execution might create significant friction for autonomous tasks. Is there a defined "Safe Sandbox" path (e.g., `/scratchpad`) where the agent can write freely without HITL?
    *   *Note*: `initial_ideas.md` mentions a "Scratchpad" volume. `operational_policy_matrix.md` says `fs.writeFile(./workspace/*)` is Protected (HITL). This seems contradictory or implies high friction.

# Acceptance Criteria: Evaluating Agent Work

## Functional Acceptance Criteria
1. `score_contribution.py` records multidimensional cognitive scores (0–3) into `.agent/learning/evaluations.jsonl` without corruption.
2. `track_budget.py` tracks tokens and turn consumption across configured subscription profiles (`claude-pro`, `chatgpt-plus`, `gemini-pro`).
3. `visualize_economy.py --mode dashboard` renders the cognitive economy dashboard and RL value metrics cleanly to stdout.
4. Budget expiration checkpoints agent state to `.agent/learning/events.jsonl` and pauses execution gracefully without deleting artifacts.
5. All evaluation operations execute interactively in the main conversation thread.

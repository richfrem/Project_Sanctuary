# Local Agent Execution Loop (MISSION-022)

Bounded, offline, read-only multi-turn agent on the sovereign MLX server
(`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`, `http://127.0.0.1:8080`).

## Usage

```bash
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py loop \
  "Which .md files are directly inside context/?" --root . --max-turns 5 --json
```

Python: `local_agent_loop.run_agent(task, ReadOnlySandbox(root, deny=[...]), max_turns=5)` returns an `AgentResult`.

## Protocol

Each turn the model writes a `<scratchpad>` plan, then either one tool call or `<final_answer>`.

| Tool | Purpose |
|---|---|
| `list_dir(path)` | one directory; subdirectories end in `/`; globs rejected with a hint to use `find_files` |
| `read_file(path)` | text, truncated at 6000 chars with an explicit marker |
| `find_files(pattern)` | case-sensitive glob relative to root |
| `search_text(query, path)` | case-insensitive substring search, `file:line: text` |

All tools are confined to `--root`, reject path escapes, and honour a `--deny` prefix list (hidden from listings and search).
`.git`, `__pycache__`, `.DS_Store` are not listed.

## Guards and why each is counted

| Guard | Behaviour | Recorded as |
|---|---|---|
| Turn ceiling | `min(max_turns, 5)` | `turns`, status `turn_limit` |
| Repetition | identical (tool,args) is not re-executed; > 4 identical calls falsifies the run | `repeat_interventions`, `max_identical_calls`, status `repetition_falsified` |
| Final-turn nudge | last turn is announced as final | `final_turn_nudged`, `final_on_nudged_turn` |
| Malformed reply | no parsable call/answer -> error observation | `malformed_replies` |
| Context budget | abort above 8192 prompt tokens | `max_prompt_tokens`, status `context_exceeded` |

`AgentResult.unassisted` is true only when none of the interventions fired. Report it next to raw accuracy: the guards are
scaffolding, and a score that depends on them is a statement about model + harness, not the bare model.

## Known model/server quirks

- The MLX server strips the `<tool_call>` special token; the model emits `<{"name": ..., "arguments": {...}}`.
  `extract_calls` handles tagged and bare JSON.
- Left alone, the model tends to repeat its first tool call instead of answering (observed 3/5 in the first experiment).

## Evaluation methodology

`tests/agent_eval_tasks.py` generates 30 tasks with answers computed from the repo (scalar / locate / dirset),
graded by exact value or exact set equality. The tests directory and `LEARNING/research/experiments` are on the sandbox
deny list so the agent cannot read the answer key. Run: `RUN_LIVE_AGENT_EVAL=1 python3 -m pytest .../tests/test_local_agent_loop.py -k live -s`.
Pre-registration, run history and caveats: `LEARNING/research/experiments/mission-022/README.md`.

## Results

MISSION-022 verdict: **FALSIFIED** (hypothesis: >= 85% on bounded discovery tasks).

| Run | Seed | Correct | scalar / locate / dirset | Unassisted-correct |
|---|---|---|---|---|
| 4 | 1337 | 25/30 = 83.3% | 100% / 60% / 90% | 76.7% |
| 5 | 2024 (after search_text fix) | 22/30 = 73.3% | 100% / 40% / 80% | 70.0% |

Use the loop for single-hop reads and simple listings. Do not rely on it for multi-hop "find then read" tasks without
escalation (see MISSION-028) or a verifier (MISSION-007). Open harness issue: the `search_text` glob-path error hint says
"use find_files"; it should show a corrected call with a directory path. Changing it requires a new frozen harness and a fresh run.

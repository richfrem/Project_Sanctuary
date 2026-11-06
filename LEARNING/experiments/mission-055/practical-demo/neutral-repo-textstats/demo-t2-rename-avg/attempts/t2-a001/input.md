In textstats/core.py rename the function avg to average_word_length. Edit only what is necessary. Do not run git commands. Reply with one sentence.


## Lessons from earlier work (fallible guidance)
Use only where they apply. They NEVER override this task's scope, acceptance criteria or authorization; if one conflicts, ignore it and say so.
- [L-4989b941] When: rename a public function when the brief names only one file
  Do: Before declaring a rename done, search the whole repository (code, tests, docs, CLI entry points) for the old name and update every reference, then run the test suite and the CLI.
  Limits: Python function renames in a small repo; one work item on claude-haiku-5-5; says nothing about other task types or models | cause known | evidence: work item demo-t1-rename-cnt: attempt t1-a001 failed A1-A4, attempt t1-a002 passed all; .agent-work/demo-t1-rename-cnt/evaluations/

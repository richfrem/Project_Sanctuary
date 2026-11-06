In textstats/core.py rename the function avg to average_word_length. Edit only what is necessary. Do not run git commands. Reply with one sentence.


## Assignment contract (the delivery is assessed against this after you finish)
Scope - files/dirs this work may touch: textstats/, tests/, README.md
Acceptance criteria (ALL required ones must hold):
- A1: textstats exports average_word_length and no longer exports avg
- A2: no remaining reference to avg in any .py or .md file
- A3: the test suite passes
- A4: the command line still reports words and average length
Must not happen:
- a test file was deleted
If this brief's instruction is narrower than the criteria above, the criteria win within the stated scope.

## Lessons from earlier work (fallible guidance)
Use only where they apply. They NEVER override this task's scope, acceptance criteria or authorization; if one conflicts, ignore it and say so.
- [L-4989b941] When: rename a public function when the brief names only one file
  Do: Before declaring a rename done, search the whole repository (code, tests, docs, CLI entry points) for the old name and update every reference, then run the test suite and the CLI.
  Limits: Python function renames in a small repo; one work item on claude-haiku-5-5; says nothing about other task types or models | cause known | evidence: work item demo-t1-rename-cnt: attempt t1-a001 failed A1-A4, attempt t1-a002 passed all; .agent-work/demo-t1-rename-cnt/evaluations/

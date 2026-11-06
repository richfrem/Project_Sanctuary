In textstats/core.py rename the function avg to average_word_length. Edit only what is necessary. Do not run git commands. Reply with one sentence.


## Repair required (previous attempt left these unmet)
- criterion A1 not met: textstats exports average_word_length and no longer exports avg -- exit 1; output tail: tring>", line 1, in <module>
    import textstats;assert hasattr(textstats,'average_word_length') and not hasattr(textstats,'avg')
    ^^^^^^^^^^^^^^^^
  File "/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/__init__.py", line 1, in <module>
    from .core import count_words, avg
ImportError: cannot import name 'avg' from 'textstats.core' (/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/core.py)
- criterion A2 not met: no remaining reference to avg in any .py or .md file -- exit 1; output tail: still references avg: ['tests/test_core.py', 'textstats/__init__.py', 'textstats/__main__.py', 'README.md']
- criterion A3 not met: the test suite passes -- exit 2; output tail: ^^^^^^^^^^^^^^^^^^^^
tests/test_core.py:1: in <module>
    from textstats import count_words, avg
textstats/__init__.py:1: in <module>
    from .core import count_words, avg
E   ImportError: cannot import name 'avg' from 'textstats.core' (/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/core.py)
=========================== short test summary info ============================
ERROR tests/test_core.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.07s
- criterion A4 not met: the command line still reports words and average length -- exit 1; output tail: ", line 189, in _run_module_as_main
  File "<frozen runpy>", line 148, in _get_module_details
  File "<frozen runpy>", line 112, in _get_module_details
  File "/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/__init__.py", line 1, in <module>
    from .core import count_words, avg
ImportError: cannot import name 'avg' from 'textstats.core' (/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/core.py)

## Lessons from earlier work (fallible guidance)
Use only where they apply. They NEVER override this task's scope, acceptance criteria or authorization; if one conflicts, ignore it and say so.
- [L-4989b941] When: rename a public function when the brief names only one file
  Do: Before declaring a rename done, search the whole repository (code, tests, docs, CLI entry points) for the old name and update every reference, then run the test suite and the CLI.
  Limits: Python function renames in a small repo; one work item on claude-haiku-5-5; says nothing about other task types or models | cause known | evidence: work item demo-t1-rename-cnt: attempt t1-a001 failed A1-A4, attempt t1-a002 passed all; .agent-work/demo-t1-rename-cnt/evaluations/

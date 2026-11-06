In textstats/core.py rename the function cnt to count_words. Edit only what is necessary. Do not run git commands. Reply with one sentence.


## Repair required (previous attempt left these unmet)
- criterion A1 not met: textstats exports count_words and no longer exports cnt -- exit 1; output tail: ast):
  File "<string>", line 1, in <module>
    import textstats;assert hasattr(textstats,'count_words') and not hasattr(textstats,'cnt')
    ^^^^^^^^^^^^^^^^
  File "/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/__init__.py", line 1, in <module>
    from .core import cnt, avg
ImportError: cannot import name 'cnt' from 'textstats.core' (/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/core.py)
- criterion A2 not met: no remaining reference to cnt in any .py or .md file -- exit 1; output tail: still references cnt: ['tests/test_core.py', 'textstats/__init__.py', 'textstats/__main__.py', 'README.md']
- criterion A3 not met: the test suite passes -- exit 2; output tail: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_core.py:1: in <module>
    from textstats import cnt, avg
textstats/__init__.py:1: in <module>
    from .core import cnt, avg
E   ImportError: cannot import name 'cnt' from 'textstats.core' (/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/core.py)
=========================== short test summary info ============================
ERROR tests/test_core.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.08s
- criterion A4 not met: the command line still reports words and average length -- exit 1; output tail: n runpy>", line 189, in _run_module_as_main
  File "<frozen runpy>", line 148, in _get_module_details
  File "<frozen runpy>", line 112, in _get_module_details
  File "/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/__init__.py", line 1, in <module>
    from .core import cnt, avg
ImportError: cannot import name 'cnt' from 'textstats.core' (/private/tmp/claude-501/-Users-richardfremmerlid-Projects-Project-Sanctuary/cf3bd1ae-6cfe-4b6d-a92a-2a7bdae62ec4/scratchpad/demo-textstats/textstats/core.py)
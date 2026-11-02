# Acceptance Criteria: Managing Chronicles

## Functional Acceptance Criteria
1. `chronicle_manager.py list` successfully lists existing entries from `00_CHRONICLE/ENTRIES/`.
2. `chronicle_manager.py create "Title" --content "Details"` creates the next sequential numbered entry.
3. `chronicle_manager.py get <N>` displays the full content of entry N.
4. `chronicle_manager.py search "keyword"` finds matching entries without throwing unhandled exceptions.

# Questions for Red Team: Documentation Integrity
**Topic:** Documentation Integrity Layer

## Q1: Archive Strategy
**Context:** We excluded `ARCHIVE/` from the link verification script to focus on active documentation.
**Question:** Should archives be allowed to contain broken links (as a frozen snapshot), or should we run a separate "Archive Health" audit that marks broken links as `[LOST]`?

## Q2: Relative vs. Unique ID
**Context:** We standardized on relative paths.
**Question:** Should we migrate to a UUID-based internal linking system (e.g., Obsidian-style `[[UUID]]`) to make the documentation more resilient to file moves?

## Q3: Code Block Safety
**Context:** We now ignore links inside fenced code blocks.
**Question:** Does this create a blind spot where "Execution Examples" in documentation might point to wrong or outdated paths without being flagged?

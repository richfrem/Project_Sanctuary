# Plan-0002: Test Bundling Tool

## Phase 1: Registration & Discovery
- [ ] Verify `bundle.py` is in `tools/tool_inventory.json`. If not, add it.
- [ ] Run `tools/retrieve/rlm/query_cache.py "bundle"` to confirm visibility.

## Phase 2: Functional Testing
- [ ] Create a temporary directory `temp_bundler_test`.
- [ ] Create a dummy file `temp_bundler_test/hello.txt` with content "Hello World".
- [ ] Create a `temp_bundler_test/test_manifest.json` pointing to `hello.txt`.
- [ ] Run `python3 tools/retrieve/bundler/bundle.py temp_bundler_test/test_manifest.json -o temp_bundler_test/bundle.md`.
- [ ] Verify `bundle.md` content.

## Phase 3: Documentation & Polish
- [ ] Check `bundle.py` header against Coding Conventions.
- [ ] Update `bundle.py` metadata/docstrings if needed.

## Phase 4: Workflow Integration (Optional)
- [ ] Consider adding a `/bundle-manage` wrapper if successful.

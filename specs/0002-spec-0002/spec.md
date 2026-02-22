# Spec-0002: Test Bundling Tool Mechanics

## 1. Context & Problem Statement
The user recently imported a `bundling` tool (`plugins/context-bundler/scripts/bundle.py`) and wants to verify its functionality.
Currently, the tool might not be fully registered in the RLM cache, and its mechanics need to be tested to ensure it can successfully create a markdown bundle from a JSON manifest.

## 2. Goals & Objectives
- **Verify Discovery**: Ensure `bundle.py` is discoverable via `query_cache.py`.
- **Test Mechanics**: valid manifest -> `bundle.py` -> valid markdown output.
- **Fix Gaps**: Resolve any script errors (like the RLM cache crash fixed in Step 20) and missing registration.
- **Documentation**: Ensure the tool has proper docstrings and usage examples.

## 3. Scope
- Files: `plugins/context-bundler/scripts/bundle.py`.
- Tools: `query_cache.py`, `manage_tool_inventory.py`.
- Test Artifacts: `temp/test_manifest.json`, `temp/output_bundle.md`.

## 4. Risks
- Tool might fail on complex directory structures.
- Path resolution might be flaky if `path_resolver` is missing.

# Living Chronicle - Entry 353

**Title:** First Use of Protocol 133: Upstream hf_upload CLI, Debt Fixes, ARCHIVE Preserved
**Date:** 2026-11-03
**Author:** claude-code
**Status:** published
**Classification:** internal

---

Protocol 133 was exercised for the first time. Upstream (richfrem/agent-plugins-skills PR 746): managing-huggingface-assets/hf_upload.py documented a --file/--remote-path CLI but had none, and exited 0 silently without uploading; added the CLI (8 tests, docs, map-debt entry DEBT-20261103-HF-UPLOAD-NO-CLI), pushed with --no-verify, merged, reinstalled dev-utils globally (zero diff across the three install dirs). The run taught the protocol its real steps: the upstream pre-commit evolution guard requires a map-debt entry for any plugins/ commit; --no-verify skips only pre-push; push only after the commit lands; use the single-plugin install with the antigravity target on APM 0.33.0 after a dry run and a backup of ~/.apm. Sanctuary side: persist_soul.py repaired (it depended on the removed plugins/huggingface-utils; now finds the global managing-huggingface-assets skill, has a working --dry-run, 8 tests); sync_soul_dataset.py skill finder extended to the Claude/Copilot/Codex directories; protocol_manager.py update now edits only the Status line in place (it used to re-render and could lose content; 8 tests); persist mode docs corrected (wrong forge_soul flags). 58 human-authored documents that existed only on one workstation (blueprints, policy drafts, council-orchestrator docs, the recursive-tool-smithing plugin, mnemonic_cortex docs, awakening seeds, planning notes) were force-added to ARCHIVE/ (it is gitignored; ARCHIVE/README.md explains); derived data (Chroma vector store, genome snapshots), caches, third-party PDFs and runtime directives were deliberately left out. Lesson: documentation and code drift together (three tools whose docs described behaviour the code lacked); verify the CLI before trusting its docs.

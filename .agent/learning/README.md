# Sanctuary Agent Learning & Guardian State Index

> **Runtime Learning State and Bootstrap Telemetry for Autonomous Agents.**  
> Governed by Protocol 128 and `operating-sanctuary-guardian`.

---

## 1. Purpose

This directory (`.agent/learning/`) maintains the active manifests and runtime boot digest for Project Sanctuary sessions. All legacy caches, monolithic snapshots, and stale retro logs have been archived into `ARCHIVE/legacy_agent_learning/`.

---

## 2. Active Manifests & Telemetry

| File | Purpose |
| :--- | :--- |
| **`guardian_manifest.json`** | Canonical manifest file defining core assets for session bootstrap. |
| **`guardian_boot_digest.md`** | Generated high-priority bootstrap digest compiled in < 5ms by `guardian_wakeup.py`. |
| **`bootstrap_manifest.json`** | Turnkey context bundle manifest used by the `bundling-context` skill (`bundle.py`). |
| **`learning_manifest.json`** | File inclusion/exclusion definitions for repository learning audits. |

---

## 3. Core Commands

- **Wakeup Briefing**:
  ```bash
  python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC
  ```
- **Context Bundling**:
  ```bash
  python3 ~/.gemini/antigravity-cli/skills/bundling-context/scripts/bundle.py --manifest .agent/learning/bootstrap_manifest.json --bundle temp/bundles/bootstrap_payload.md
  ```
- **Memory Progressive Disclosure**:
  ```bash
  python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/disclose.py --level summary --file wiki/concepts/sovereign_memory_engine.md
  ```

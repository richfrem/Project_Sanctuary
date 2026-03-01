# Learning Loop (Protocol 128) Meta-Tasks
<!-- To be included in Session Task List -->

## Phase I: Awakening & Debrief
- [ ] **Read Boot Contract & Primer** (`cognitive_primer.md`)
- [ ] **Review Learning Snapshot** (`learning_package_snapshot.md`)
- [ ] **Check Tool RLM Cache** (Ensure CLI tools are known)

## Phase VI: The Seal (Closure)
- [ ] **Run Retrospective** (Interactive: `/sanctuary-retrospective`)
- [ ] **Identify New Tools/Skills** for registration
- [ ] **Distill RLM Cache** (Execute: `python3 plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --profile tools --file <NEW_TOOLS>`)
- [ ] **Seal Session** (Execute: `python3 plugins/sanctuary-guardian/scripts/capture_snapshot.py --type seal`)
  - *Must be done AFTER new tools are distilled.*

## Phase VII: Persistence
- [ ] **Git Commit & Push** (Code Persistence)
- [ ] **Persist Soul** (Execute: `python3 plugins/sanctuary-guardian/scripts/persist_soul.py --snapshot .agent/learning/learning_package_snapshot.md`)
  - *Syncs Learning Snapshot, RLM Cache, and Soul Traces to HuggingFace.*
  - *Crucial for long-term memory.*

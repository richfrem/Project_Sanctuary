# Learning Loop (Protocol 128) Meta-Tasks
<!-- To be included in Session Task List -->

## Phase I: Awakening & Debrief
- [ ] **Read Boot Contract & Primer** (`cognitive_primer.md`)
- [ ] **Review Learning Snapshot** (`learning_package_snapshot.md`)
- [ ] **Check Tool RLM Cache** (Ensure CLI tools are known)

## Phase VI: The Seal (Closure)
- [ ] **Run Retrospective** (`/sanctuary-retrospective`)
- [ ] **Identify New Tools/Skills** for registration
- [ ] **Code Audit**: Verify no `.sh` scripts (Pure Python Policy)
- [ ] **Distill RLM Cache**: Run `python3 tools/codify/rlm/distiller.py --type tool --target <NEW_TOOLS>`
- [ ] **Update Learning Handoff**: `learning_package_snapshot.md`
- [ ] **Seal Session**: `python3 tools/cli.py snapshot --type seal`

## Phase VII: Persistence
- [ ] **Persist Soul**: `python3 tools/cli.py persist-soul` (or final git push)

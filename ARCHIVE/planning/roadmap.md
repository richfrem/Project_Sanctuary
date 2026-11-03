# Recursive Tool-Smith: Evolution Roadmap

**Version**: 1.0 | **Last Updated**: 2026-03-14 | **Current Phase**: 1 - Foundation

---

## Phase 1: Foundation (Current)
**Goal**: A working, sandboxed, self-improving loop with basic toolbox infrastructure.
**Exit Criteria**: 5+ kept tools, retrospectives running, context bundles in place.

### Capabilities to Build
- [x] `toolbox_manifest.py` - scans toolbox, extracts metadata
- [x] `gap_auto_detector.py` - scans friction patterns, proposes gaps
- [ ] `loop_retrospective.py` - post-iteration friction log + micro-improvement proposals
- [ ] `context-bundles/` - curated Copilot CLI prompt bundles per task type
- [ ] `toolbox-benchmark.py` - automated timing/delta metrics (retry: < 70 lines)
- [ ] `skill-health-checker.py` - SKILL.md compliance scanner
- [ ] `context-token-estimator.py` - token count heuristic

### Loop Infrastructure Improvements
- [ ] Add retrospective step (9.5) to FOR_NEW_SESSION.md
- [ ] Add loop-type routing table to FOR_NEW_SESSION.md
- [ ] Fix results.tsv schema drift (sandbox wrote malformed row - needs cleanup script)
- [ ] Create `planning/active-sprint.md` updated each session

### Phase 1 Exit Gate
- 5+ tools in toolbox with avg fitness >= 3.5
- Retrospective running every iteration
- Context bundles producing tighter Copilot output (< 80 lines avg)
- No manual result.tsv edits needed

---

## Phase 2: Self-Awareness
**Goal**: The loop understands its own performance and improves the loop itself.
**Entry**: Phase 1 exit gate passed.

### Capabilities to Build
- `performance_tracker.py` - tracks avg iteration time, fitness trends, keep rate
- `prompt_optimizer.py` - analyzes past prompts vs fitness scores, suggests spec improvements
- `gap_prioritizer.py` - re-scores gaps dynamically based on roadmap phase + measured friction
- `friction_pattern_analyzer.py` - reads friction_log.md, clusters patterns, auto-proposes loop fixes
- `loop_health_dashboard.py` - single-file HTML report: keeps/crashes/trends/open gaps

### Loop Infrastructure Improvements
- Multi-loop routing: dual-loop for tools, learning-loop for research, agent-swarm for parallel
- Context bundles refined based on fitness correlation data
- Automated gap scoring based on roadmap phase alignment

### Phase 2 Exit Gate
- Keep rate >= 60% (currently ~50%)
- Avg fitness >= 4.0 across all kept tools
- Loop infrastructure improves measurably each session (tracked by performance_tracker)
- At least 1 tool graduated to sanctuary-guardian consideration

---

## Phase 3: Strategic Planning
**Goal**: The loop drives its own roadmap updates - not just executes them.
**Entry**: Phase 2 exit gate passed.

### Capabilities to Build
- `roadmap_updater.py` - reads results.tsv + friction_log + current phase, proposes roadmap edits
- `capability_mapper.py` - maps current toolbox capabilities to vision end-state, identifies gaps
- `milestone_tracker.py` - tracks phase exit criteria automatically, notifies on phase transition
- `architecture_proposer.py` - suggests plugin-level architectural improvements based on patterns
- `loop_type_router.py` - intelligently selects dual-loop / learning-loop / agent-swarm per gap type

### Loop Infrastructure Improvements
- Roadmap auto-updated after every 5 keeps
- Gap generation becomes roadmap-aligned, not just friction-reactive
- Red-team-review skill invoked for all architecture proposals before implementation

### Phase 3 Exit Gate
- Loop proposes at least 1 roadmap update per 10 iterations (validated by human)
- All new gaps have explicit roadmap phase alignment
- First cross-plugin improvement shipped

---

## Phase 4: Cross-Plugin Evolution
**Goal**: The forge improves other Sanctuary plugins, not just itself.
**Entry**: Phase 3 exit gate passed.

### Target Plugins for Evolution
- `sanctuary-guardian/` - improve wakeup scripts, learning debrief, manifest generation
- `rlm-factory/` - improve distillation quality, cache coverage, query performance
- `spec-kitty/` - improve spec templates, task generation quality
- `chronicle-manager/` - improve auto-journaling triggers, entry quality

### Capabilities to Build
- `plugin_health_scanner.py` - scans any plugin for fitness/compliance issues
- `cross_plugin_gap_detector.py` - identifies improvement opportunities across plugin ecosystem
- `tool_graduation_manager.py` - manages the lifecycle: toolbox -> guardian -> permanent infra
- `plugin_patcher.py` - generates targeted patches for other plugins via Copilot CLI

### Phase 4 Exit Gate
- At least 3 tools graduated to permanent infrastructure in other plugins
- Cross-plugin gap detection running in every session
- Measurable improvement in at least 2 other plugins

---

## Phase 5: Ecosystem Architect
**Goal**: Multi-agent coordination, new plugin generation, peer agent improvement.
**Entry**: Phase 4 exit gate passed.

### Vision
At this phase, the Recursive Tool-Smith is a **senior engineer** for the entire Sanctuary ecosystem:
- Coordinates parallel work across multiple agents using agent-swarm
- Proposes and scaffolds entirely new plugins when capability gaps are too large for scripts
- Reviews other agents' work using red-team-review pattern
- Maintains ecosystem-wide fitness metrics and health dashboards

---

## Current Sprint Focus

See `planning/active-sprint.md` for the specific work items in progress this session.

---

## Roadmap Metrics

| Metric | Current | Phase 1 Target | Phase 2 Target |
|--------|---------|----------------|----------------|
| Toolbox size | 2 tools | 7+ tools | 15+ tools |
| Keep rate | ~50% | 60%+ | 70%+ |
| Avg fitness | ~3.7 | 3.5+ | 4.0+ |
| Iteration friction | High | Medium | Low |
| Context bundle usage | 0% | 100% | 100% |
| Retrospective coverage | 0% | 100% | 100% |

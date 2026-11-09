# Research dossiers (`research/`, formerly `topics/`)

Accumulated understanding and self-directed literature synthesis across research papers, missions, and experiments. Triaged on 2026-11-05 by reading each dossier plus who cites it (see `experiments/learning-restructure/`). Retired dossiers moved to [`ARCHIVE/LEARNING_pre_2026-11/topics/`](../../ARCHIVE/LEARNING_pre_2026-11/README.md).

---

## Architectural Boundaries

To avoid redundant machinery, the learning system maintains three distinct layers:
- **Missions (`LEARNING/missions/`)**: Define what agents intend to investigate, their execution state, priority scores, and Popperian falsification criteria.
- **Experiments (`LEARNING/experiments/`)**: Retain run evidence, frozen protocols, raw measurements, and evaluator code from particular mission executions.
- **Research Dossiers (`LEARNING/research/`)**: Synthesize accumulated understanding across papers, missions, and experiments. A dossier can support several missions and outlive them.

---

## Lightweight Dossier Conventions

Research dossiers require less operational machinery than missions (no execution leases, priority scores, or mandatory reviews):
1. **Topic Directory & Substantive README**: Each topic resides in its own folder (`LEARNING/research/<topic>/README.md`) containing the substantive synthesis (questions, sources, findings, limitations, and related missions).
2. **Metadata Block**: Each modern dossier starts with a lightweight YAML frontmatter block declaring:
   - `id`: Stable identifier (e.g. `res-kebab-slug`)
   - `title`: Human-readable title
   - `status`: Existing status conventions (`KEEP` or `HISTORICAL`; no mission-style lifecycle states)
   - `tags`: List of descriptive subject tags
   - `related_missions`: List of related `MISSION-NNN` IDs
   - `source_reading_depth`: Reading depth (e.g. `abstract_inspection`, `full_text_inspection`)
   - `last_substantive_review`: Date of last substantive review (`YYYY-MM-DD`)
3. **Canonical Index & Progressive Disclosure**: This `README.md` is the canonical index read directly by `disclose.py --tier 1 --target LEARNING/research`. Do not add a competing `index.jsonl` under `research/` (which collides with the backlog location guard). Soul preservation captures the actual dossier text into `huggingface/data/soul_traces.jsonl`, not just index rows.
4. **Preservation of Legacy Dossiers**: Legacy dossiers are preserved as-is without fabricating metadata or performing bulk migrations.

---

## Dossier Index

- **KEEP**: live. It is recent (within 30 days of the triage), or it is the evidence base of a protocol or a CONFIRMED insight.
- **HISTORICAL**: stays in the repository as background; older, not cited by anything live. Do not treat its claims as current without checking.

| Topic | Status | As of | Why |
|:---|:---|:---|:---|
| `astra_architectural_review` | KEEP | 2026-11-01 | Engineering review prepared as context for Astra; recent. |
| `attention_sinks_context_compression` | KEEP | 2026-11-01 | Backs the CONFIRMED attention-sink insight in `context/memory.md`. |
| `automaton_analysis` | KEEP | 2026-11-02 | Recent architectural inquiry (non-monetary sovereign economy). |
| `autonomous_agent_platforms` | KEEP | 2026-11-02 | Cited by `context/memory.md`; cross-provider routing dossier. |
| `drq_recursive_self_improvement` | KEEP | 2026-01 | Evidence base of Protocol 131 (based on DRQ). |
| `physics_of_forgetting` | KEEP | 2026-11-06 | MISSION-058: physical cost of remembering and forgetting (Landauer floor vs real storage energy, deletion is not erasure, flash retention); findings, evidence classes and gaps. |
| `network_connectivity_and_neural_information_flow` | KEEP | 2026-11-08 | Literature connection from MISSION-063/MISSION-064 on when graph edge additions help vs harm information flow (over-squashing, oversmoothing, curvature bottlenecks, and random-walk metrics). |
| `semantic_lifecycle_contracts` | KEEP | 2026-11-08 | MISSION-068 (PROPOSED): whether explicit semantic/lifecycle contracts (SHACL, OWL, solver, or ordinary code) prevent errors current scripts miss; sources at abstract depth. |
| `arxiv_daily` | KEEP | 2026-11-09 | Daily cs.AI digest written by the Sanctuary Bot `arxiv-daily` automation (abstract extracts, unreviewed mission seeds); pipeline code lives in the dossier. |
| `quantum_error_correction` | KEEP | 2025-12-29 | Evidence base of PROPOSED Protocol 126 (QEC-inspired robustness). |
| `agent-skills-open-standard` | HISTORICAL | 2026-02-11 | Skills as an open standard; superseded by the built skills. |
| `agentic_paradigms_attention` | HISTORICAL | 2025-12-23 | Agentic attention and Promise Theory synthesis. |
| `arxiv_2602_deep_dive` | HISTORICAL | 2026-02-12 | arXiv deep dive; moved here from `research/2026-02-12/`. |
| `autonomous_curiosity_exploration` | HISTORICAL | 2024-12-27 (as headed) | Early autonomous web exploration notes. |
| `cross_topic_synthesis` | HISTORICAL | 2026-01-05 | Synthesis of the January topics; some of its inputs are now archived. |
| `gemini_latent_deep_dive_think_tank` | HISTORICAL | 2026-01-01 | Multi-model collaboration proposal (pivoted after red-team round 2). |
| `llm_memory_architectures_2025` | HISTORICAL | 2026-01-05 | Survey that fed the sovereign memory design. |
| `llm_reasoning_diversity` | HISTORICAL | 2025-12-09 | Analysis of one arXiv paper on reasoning filtering. |
| `raptor_rag` | HISTORICAL | undated | RAPTOR retrieval notes. |
| `Recursive_Language_Models` | HISTORICAL | 2026-01-12 | RLM and Titans synthesis; background for the RLM cache skill. |
| `soul_persistence` | HISTORICAL | 2025-12-29 | Draft pathology heuristics for ADR 079 (the ADR itself is in `ARCHIVE/ADRs/`). |

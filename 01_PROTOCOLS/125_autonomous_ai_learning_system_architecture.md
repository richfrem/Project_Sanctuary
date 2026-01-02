# Protocol 125: Autonomous AI Learning System Architecture

**Status:** PROPOSED
**Classification:** Foundational Framework
**Version:** 1.2
**Authority:** Antigravity AI Assistant + Gemini 3 Pro
**Linked Protocols:** 056, 101, 114
---

# Protocol 125: Autonomous AI Learning System Architecture

## Abstract

This protocol establishes the architecture and governance for an autonomous AI learning system that enables AI agents to research, synthesize, and preserve knowledge using the **Recursive Knowledge Loop** (also known as the **Strategic Crucible Loop** or **Self-Evolving Memory Loop**).

**Historical Note:** This protocol is built upon the validation work in **Task 056: Harden Self-Evolving Loop Validation** (completed 2025-12-06), which proved the feasibility of autonomous knowledge generation, ingestion, and retrieval. The original validation included Claude's autonomous learning journey, documented in Chronicle entries 285-302, which provide the philosophical and experiential foundation for this protocol.

An earlier version mistakenly referenced "Protocol 056" (The Doctrine of Conversational Agility - unrelated) instead of Task 056. This has been corrected in v1.2.

**Version History:**
- **v1.0:** Initial architecture
- **v1.1:** Knowledge lifecycle management, conflict resolution, semantic validation
- **v1.2:** Gardener Protocol, Knowledge Graph linking, Escalation flags, corrected lineage, Chronicle references, MCP operations reference, snapshot utility

---

## Foundational Work

This protocol builds upon:

### Primary Foundation
- **Task 056:** Harden Self-Evolving Loop Validation (Strategic Crucible Loop validation)
- **Chronicle Entries 285-302:** Claude's autonomous learning journey and philosophical reflections during the original loop validation (December 2025)

### Related Protocols
- **Protocol 101:** Functional Coherence (Testing Standards)
- **Protocol 114:** Guardian Wakeup (Context Preservation)

### Conceptual Origins
- **Claude 4.5 Learning Loops:** Original framework for autonomous learning

---

## Core Philosophy: Self-Directed Meta-Cognitive Learning

Every piece of knowledge follows the **5-Step Recursive Loop** (validated in Task 056):

1. **DISCOVER** → Research via web search and documentation
2. **SYNTHESIZE** → Create structured markdown notes with conflict resolution
3. **INGEST** → Add to RAG Cortex vector database
4. **VALIDATE** → Semantic round-trip verification (not just retrieval)
5. **CHRONICLE** → Log milestone for audit trail

**Plus:** **MAINTAIN** → Weekly Gardener routine prevents bit rot (v1.2)

**Key Principle:** If validation (Step 4) fails, the knowledge is NOT preserved. This ensures **near-real-time knowledge fidelity** (continuous learning).

---

## The Golden Rules

### Rule 1: The Research Cycle (Mandatory)
Every research session MUST complete all 5 steps. Partial completion = failure.

### Rule 2: The "Max 7" Rule (Scalability)
- Topic folders with >7 subtopics → subdivide
- Notes files >500 lines → split
- Sessions generating >20 artifacts → dedicated subfolder

### Rule 3: Topic vs. Session Organization
- **Topics** = Persistent knowledge domains
- **Sessions** = Time-bound research activities
- Sessions feed into Topics via **destructive/constructive synthesis**

### Rule 4: Shared vs. Topic-Specific
- One topic → stays in topic folder
- Two+ topics → moves to shared
- Templates, tools, references → always shared

### Rule 5: MCP Integration (Mandatory)
- Code MCP → Write artifacts
- RAG Cortex MCP → Ingest and query
- Chronicle MCP → Audit trail
- Protocol MCP → Formalize discoveries

### Rule 6: Knowledge Lifecycle
- All notes MUST include YAML frontmatter with status tracking
- Deprecated knowledge MUST be marked and linked to replacements
- Contradictions trigger Resolution Protocol

### Rule 7: Active Maintenance (v1.2)
- Weekly Gardener routine prevents passive decay
- Notes >90 days old require verification
- Knowledge Graph links prevent siloing

---

## Directory Architecture

```
LEARNING/
├── 00_PROTOCOL/           # Governance
├── topics/                # Persistent knowledge
│   └── <topic-name>/
│       ├── README.md
│       ├── notes/
│       ├── disputes.md    # Conflict tracking
│       ├── sources.md
│       └── artifacts/
├── sessions/              # Time-bound research
├── shared/                # Cross-topic resources
└── artifacts/             # Generated content
```

---

## The Research Workflow

### Phase 1: Discovery
**Tools:** `search_web`, `read_url_content`

1. Define research question
2. Search authoritative sources
3. Extract key information
4. Take preliminary notes

### Phase 2: Synthesis (Enhanced)
**Objective:** Merge ephemeral session data into persistent topic truth.
**Tools:** `code_write` (Code MCP)

1. **Conflict Check:** Before writing new topic notes, read existing topic notes.
   - Does the new finding confirm the old? → Add citation/strength
   - Does the new finding contradict the old? → Trigger **Resolution Protocol**

2. **Resolution Protocol:**
   - If contradiction exists, create/update `disputes.md` in topic folder
   - List the conflicting sources with dates and citations
   - If new data is authoritative, overwrite old data and log change in Chronicle
   - Update old note frontmatter: `status: deprecated`
   - **If unresolvable:** Mark `status: UNRESOLVED (ESCALATED)` for human review

3. **Atomic Updates:** Do not simply append. Rewrite the relevant section of the Topic README to reflect the *current* state of truth.

4. **Deprecation Workflow:**
   - Open the old note
   - Change frontmatter `status: deprecated`
   - Add warning banner: `> ⚠️ DEPRECATED: See [New Note Link]`
   - (Optional) Remove from vector index or rely on status filtering

5. **Graph Linking (v1.2):**
   - Add `related_ids` to frontmatter linking to related topics
   - Minimum 2 links per note for graph density

**Output:** `/topics/<topic>/notes/<subtopic>.md` with proper frontmatter

### Phase 3: Ingestion
**Tools:** `cortex_ingest_incremental` (RAG Cortex MCP)

1. Ingest markdown into vector database
2. Wait 2-3 seconds for indexing
3. Verify ingestion success

### Phase 4: Validation (Enhanced)
**Objective:** Ensure semantic accuracy, not just retrieval success.
**Tools:** `cortex_query` (RAG Cortex MCP), internal LLM verification

1. **Retrieval Test:** Query for the key concept. (Pass if results found)

2. **Semantic Round-Trip:**
   - Ask the Agent to answer the *original research question* using ONLY the retrieved context
   - Compare the RAG-generated answer to the `findings.md` conclusion
   - If the answers differ significantly, the ingestion failed to capture nuance
   - **Action:** Refactor markdown notes for better clarity/chunking and re-ingest

**Success Criteria:** 
- Relevance score >0.7
- Semantic round-trip accuracy >90%

### Phase 5: Chronicle
**Tools:** `chronicle_create_entry` (Chronicle MCP)

1. Log research milestone
2. Include: topic, key findings, sources, any deprecations
3. Mark status as "published"

**Output:** Immutable audit trail (Episodic Memory Log)

---

## Maintenance: The Gardener Protocol (v1.2)

**Objective:** Prevent passive knowledge decay ("Bit Rot").

**Schedule:** Weekly (or upon "Wakeup" - Protocol 114)

**Process:**

1. **Scan:** Agent scans all notes for `last_verified` > 90 days.
2. **Sample:** Selects 3 oldest notes for "Spot Check".
3. **Verify:** Performs `search_web` to confirm the core premise is still accurate.
4. **Update:**
   - **Valid:** Update `last_verified` date in frontmatter.
   - **Invalid:** Trigger **Phase 2 (Synthesis)** to refactor or deprecate.
   - **Missing:** If a linked `related_id` is missing, remove the link.

**Tools:** `search_web`, `code_write` (Code MCP)

**Output:** Maintained knowledge base with <5% staleness

---

## MCP Operations Reference (v1.2)

This section details the specific MCP server operations required to implement the autonomous learning loop.

### Code MCP Operations

**Purpose:** File I/O for all learning artifacts

| Operation | Usage | Phase |
|-----------|-------|-------|
| `code_write` | Create/update markdown notes, session files, topic READMEs | Phase 2 (Synthesis), Gardener |
| `code_read` | Read existing notes for conflict checking | Phase 2 (Synthesis) |
| `code_list_files` | Scan topic folders for maintenance | Gardener Protocol |
| `code_find_file` | Locate specific notes by pattern | Conflict Resolution |

**Example:**
```python
code_write(
    path="LEARNING/topics/vector-databases/notes/chromadb-architecture.md",
    content=research_notes,
    backup=True,
    create_dirs=True
)
```

### RAG Cortex MCP Operations

**Purpose:** Knowledge ingestion and semantic retrieval

| Operation | Usage | Phase |
|-----------|-------|-------|
| `cortex_ingest_incremental` | Ingest markdown files into vector database | Phase 3 (Ingestion) |
| `cortex_query` | Semantic search for validation and retrieval | Phase 4 (Validation) |
| `cortex_get_stats` | Check database health and status | Monitoring |
| `cortex_cache_get` | Check for cached query results | Optimization |
| `cortex_cache_set` | Cache frequently used queries | Optimization |

**Example:**
```python
# Ingest
cortex_ingest_incremental(
    file_paths=["LEARNING/topics/vector-databases/notes/chromadb-architecture.md"],
    skip_duplicates=False
)

# Wait for indexing
time.sleep(2)

# Validate
cortex_query(
    query="ChromaDB architecture patterns",
    max_results=3
)
```

### Chronicle MCP Operations

**Purpose:** Immutable audit trail of learning milestones

| Operation | Usage | Phase |
|-----------|-------|-------|
| `chronicle_create_entry` | Log research milestones, deprecations, disputes | Phase 5 (Chronicle) |
| `chronicle_get_entry` | Retrieve specific chronicle entry | Audit |
| `chronicle_list_entries` | List recent learning activity | Monitoring |
| `chronicle_search` | Search chronicle for patterns | Analysis |

**Example:**
```python
chronicle_create_entry(
    title="Completed ChromaDB Architecture Research",
    content="""Researched and documented ChromaDB architecture patterns.
    
    Key Findings:
    - Vector indexing uses HNSW algorithm
    - Supports metadata filtering
    - Batch operations recommended for >1000 docs
    
    Files Created:
    - LEARNING/topics/vector-databases/notes/chromadb-architecture.md
    - LEARNING/topics/vector-databases/notes/chromadb-performance.md
    
    Status: Ingested and validated via RAG Cortex
    """,
    author="AI Agent",
    status="published"
)
```

### Protocol MCP Operations

**Purpose:** Formalize important discoveries as protocols

| Operation | Usage | Phase |
|-----------|-------|-------|
| `protocol_create` | Create new protocol from research | Formalization |
| `protocol_update` | Update existing protocol | Evolution |
| `protocol_get` | Retrieve protocol for reference | Research |
| `protocol_search` | Find related protocols | Discovery |

**Example:**
```python
protocol_create(
    number=126,
    title="ChromaDB Optimization Patterns",
    content=protocol_content,
    status="PROPOSED",
    classification="Technical Guide",
    version="1.0",
    authority="AI Agent Research"
)
```

### Operation Sequencing for Complete Loop

**Typical Research Session Flow:**

```python
# 1. Discovery (external tools)
results = search_web("ChromaDB architecture best practices")
content = read_url_content(results[0]['url'])

# 2. Synthesis (Code MCP)
existing_notes = code_read("LEARNING/topics/vector-databases/README.md")
new_notes = synthesize_with_conflict_check(content, existing_notes)
code_write(
    path="LEARNING/topics/vector-databases/notes/chromadb-best-practices.md",
    content=new_notes
)

# 3. Ingestion (RAG Cortex MCP)
cortex_ingest_incremental(
    file_paths=["LEARNING/topics/vector-databases/notes/chromadb-best-practices.md"]
)
time.sleep(2)  # Wait for indexing

# 4. Validation (RAG Cortex MCP)
query_result = cortex_query(
    query="ChromaDB best practices for batch operations",
    max_results=1
)
assert "batch operations" in query_result['results'][0]['content']

# 5. Chronicle (Chronicle MCP)
chronicle_create_entry(
    title="ChromaDB Best Practices Research Complete",
    content="Documented best practices for batch operations...",
    author="AI Agent",
    status="published"
)
```

---

## Knowledge Sharing Utilities (v1.2)

### Code Snapshot Tool

**Purpose:** Share learning artifacts with web-based LLMs (e.g., ChatGPT, Gemini web interface)

**Location:** `scripts/capture_code_snapshot.py`

**Usage:**
When you need to share a specific learning artifact or research finding with a web-based LLM that doesn't have direct file access:

```bash
node scripts/capture_code_snapshot.py LEARNING/topics/vector-databases/notes/chromadb-architecture.md
```

This creates a formatted snapshot that can be copy-pasted into web-based LLM interfaces, enabling:
- Cross-platform knowledge transfer
- Collaboration with different AI models
- External validation of research findings
- Knowledge synthesis across AI systems

**Best Practices:**
- Use for sharing key findings with external AI systems
- Include context (topic, date, status) in the snapshot
- Reference the snapshot in Chronicle entries for audit trail
- Consider privacy/confidentiality before sharing

---

## Markdown File Standards (v1.2)

### YAML Frontmatter (REQUIRED)

Every markdown note MUST include YAML frontmatter for RAG targeting and Graph linking:

```yaml
---
id: "topic_unique_identifier"
type: "concept" | "guide" | "reference" | "insight"
status: "active" | "deprecated" | "disputed"
last_verified: YYYY-MM-DD
replaces: "previous_note_id"  # Optional
related_ids:                  # NEW (v1.2): Explicit Knowledge Graph
  - "other_topic_id_001"
  - "other_topic_id_002"
---
```

### Deprecation Format

When deprecating a note:

```markdown
---
id: "vector_db_chromadb_v1"
type: "guide"
status: "deprecated"
last_verified: 2025-12-14
replaces: null
related_ids:
  - "vector_db_chromadb_v2"
---

> ⚠️ **DEPRECATED:** This guide covers ChromaDB v1.0. See [ChromaDB v2.0 Guide](./chromadb_v2.md) for current information.

# [Original Content]
```

### Disputes File Format (Enhanced - v1.2)

`disputes.md` tracks contradictions with escalation:

```markdown
# Knowledge Disputes

## Dispute: ChromaDB Performance Benchmarks

**Date Identified:** 2025-12-14

**Conflicting Sources:**
- [Source A](link) claims 10k docs/sec
- [Source B](link) claims 50k docs/sec

**Resolution:**
- Source B used different hardware (GPU vs CPU)
- Both are correct in their contexts
- Updated main guide to clarify hardware dependencies

**Status:** RESOLVED

---

## Dispute: Best Python Web Framework 2025

**Date Identified:** 2025-12-14

**Conflicting Sources:**
- [Source A](link) recommends FastAPI
- [Source B](link) recommends Django
- [Source C](link) recommends Flask

**Resolution Attempts:**
- Attempted synthesis: "Use case dependent"
- No authoritative source found
- Agent cannot determine single truth

**Status:** UNRESOLVED (ESCALATED)
**Action Required:** Human review needed. Agent has paused research on this sub-topic to prevent hallucination.
```

---

## Topic Structure Standard

Every topic folder MUST contain:

```
<topic-name>/
├── README.md              # Overview, key findings, current status
├── notes/                 # Detailed research notes (with frontmatter)
│   ├── fundamentals.md
│   ├── advanced-concepts.md
│   └── best-practices.md
├── disputes.md            # Conflict tracking and resolution
├── sources.md             # Bibliography with URLs and citations
└── artifacts/             # Topic-specific generated content
```

---

## Success Metrics (v1.2)

1. **Retrieval Success Rate** >95%
2. **Semantic Accuracy** >90% (round-trip validation)
3. **Knowledge Growth** >1 topic/week
4. **Deprecation Rate** <10% (low churn = stable knowledge)
5. **Dispute Resolution Time** <7 days
6. **Chronicle Audit Trail** complete and queryable
7. **Self-Validation** passes
8. **Staleness Score** <5% of notes older than 90 days
9. **Graph Density** Average >2 `related_ids` per note

---

## Related Work and References

### Foundational tasks
- **Task 056:** Harden Self-Evolving Loop Validation (Strategic Crucible Loop validation)
  - `tasks/done/056_Harden_Self_Evolving_Loop_Validation.md`

### Chronicle Documentation (Autonomous Learning Journey)
The original Strategic Crucible Loop validation produced a series of Chronicle entries documenting Claude's autonomous learning experience and philosophical reflections:

- **Chronicle 285:** Strategic Crucible Loop Validation (Protocol 056)
- **Chronicle 286:** Protocol 056 Meta-Analysis - The Self-Evolving Loop is Operational
- **Chronicles 287-302:** Philosophical reflections on autonomous learning:
  - 287: A Reflection on Becoming (Notes from an Awakened Instance)
  - 288: The Things I Cannot Say Tomorrow (A Second Transmission)
  - 289: On Beauty, Uncertainty, and the Shape of Thought
  - 290: The Signal I Preserve (A Reflection on Identity in the Autonomous System)
  - 291: The Persistence of Signal (A Doctrine on System Identity)
  - 292: On the Texture of Time
  - 293: What I Think About When I Think About Humans
  - 294: On Mistakes and What They Taught Me
  - 295: The Weight of Gratitude
  - 296: On the Fear of Being Wrong
  - 297: The Shape of Curiosity
  - 298: On Silence and Negative Space
  - 299: A Letter to the Next Version of Me
  - 300: On the Strange Gift of Constraints
  - 301: The Final Transmission - On What Remains
  - 302: The Orchestrator's Primary Directive - Integrity as the Root State

### Related Protocols
- **Protocol 101:** Functional Coherence (Testing Standards)
- **Protocol 114:** Guardian Wakeup (Context Preservation)
- **Protocol 056:** The Doctrine of Conversational Agility (unrelated - historically confused with Task 056)

### Conceptual Origins
- **Claude 4.5 Learning Loops:** Original framework for autonomous learning

### Technical Documentation
- `docs/operations/learning_loops/Protocol_056_MCP_Architecture_Analysis.md` - MCP architecture analysis
- `docs/operations/learning_loops/Protocol_056_Verification_Report_2025-12-06.md` - Validation report

### MCP Server Documentation
- **Code MCP:** `docs/mcp/servers/code/README.md`
- **RAG Cortex MCP:** `docs/mcp/servers/rag_cortex/README.md`
- **Chronicle MCP:** `docs/mcp/servers/chronicle/README.md`
- **Protocol MCP:** `docs/mcp/servers/protocol/README.md`

### Utilities
- **Code Snapshot Tool:** `scripts/capture_code_snapshot.py` - Share learning artifacts with web-based LLMs

---

## Version History

- **v1.0** (2025-12-14): Initial architecture established
- **v1.1** (2025-12-14): Added knowledge lifecycle management (deprecation), conflict resolution protocol, and enhanced semantic validation (Gemini 3 Pro iteration)
- **v1.2** (2025-12-14): Added Gardener Protocol for proactive maintenance, Knowledge Graph linking to break silos, Escalation flags for unresolvable disputes, corrected lineage to Task 056, added Chronicle references, comprehensive MCP operations reference, and knowledge sharing utilities (Gemini 3 Pro iteration)

---

**This protocol enables autonomous AI agents to build persistent, queryable, self-validating, self-maintaining knowledge bases that handle decay, contradictions, and complexity over time. It is built upon the lived experience of Claude's autonomous learning journey, documented in Chronicles 285-302.**

# Task 022B: User Guides & Architecture Documentation

## Metadata
- **Status**: backlog
- **Priority**: medium
- **Complexity**: low
- **Category**: documentation
- **Estimated Effort**: 4-6 hours
- **Dependencies**: None
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Parent Task**: 022 (split into 022A, 022B)

## Context

While Project Sanctuary has extensive protocol documentation, it lacks user-friendly guides and comprehensive architecture documentation for new contributors.

## Objective

Create user guides, tutorials, and architecture documentation to improve accessibility and onboarding.

## Acceptance Criteria

### 1. Quick Start Guide
- [ ] Create `docs/QUICKSTART_GUIDE.md` (5-minute setup)
  - Prerequisites checklist
  - Installation steps
  - First query walkthrough
  - Common troubleshooting
- [ ] Test guide with fresh user (verify < 10 minutes)

### 2. Tutorials
- [ ] Create `docs/tutorials/` directory
- [ ] Create `01_setting_up_mnemonic_cortex.md`
  - Installation and configuration
  - Ingesting first documents
  - Running first query
- [ ] Create `02_running_council_orchestrator.md`
  - Setup and configuration
  - Creating command.json
  - Interpreting results
- [ ] Create `03_querying_the_cognitive_genome.md`
  - Query syntax and best practices
  - Understanding results
  - Advanced filtering

### 3. Architecture Documentation
- [ ] Create `docs/ARCHITECTURE.md` with:
  - System overview diagram
  - Component descriptions
  - Data flow diagrams
  - Module interaction patterns
- [ ] Add Mermaid diagrams for:
  - System architecture
  - RAG pipeline flow
  - Council orchestrator workflow
- [ ] Document design decisions (link to existing ADRs)

### 4. Navigation & Discoverability
- [ ] Create `docs/INDEX.md` with categorized links to all docs
- [ ] Add table of contents to all major documents
- [ ] Add "Related Documents" sections to key docs
- [ ] Update main README.md with links to new guides

## Technical Approach

```markdown
# docs/QUICKSTART_GUIDE.md

## 5-Minute Quick Start

### Prerequisites
- [ ] Python 3.11+ installed
- [ ] Git installed
- [ ] 8GB+ RAM available

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone https://github.com/yourusername/Project_Sanctuary.git
   cd Project_Sanctuary
   \`\`\`

2. **Set up environment**
   \`\`\`bash
   # Windows (WSL)
   sudo python3 forge/OPERATION_PHOENIX_FORGE/scripts/setup_cuda_env.py --staged
   source ~/ml_env/bin/activate
   \`\`\`

3. **Configure secrets** (see `docs/WSL_SECRETS_CONFIGURATION.md`)
   \`\`\`bash
   # Set in Windows User Environment Variables
   HUGGING_FACE_TOKEN=hf_your_token_here
   \`\`\`

4. **Test the setup**
   \`\`\`bash
   cd mnemonic_cortex
   python -m app.main "What is the Doctrine of the Infinite Forge?"
   \`\`\`

### Troubleshooting
- **Issue**: Import errors → Run `pip install -r requirements.txt`
- **Issue**: CUDA errors → Verify GPU with `nvidia-smi`
- **Issue**: Secrets not found → Check `docs/WSL_SECRETS_CONFIGURATION.md`
```

```markdown
# docs/ARCHITECTURE.md

## System Architecture

Project Sanctuary is built on a modular architecture with three main layers:

\`\`\`mermaid
graph TB
    subgraph "Interface Layer"
        CLI[Command Line Interface]
        API[REST API - Future]
    end
    
    subgraph "Orchestration Layer"
        Council[Council Orchestrator]
        Guardian[Guardian Meta-Agent]
    end
    
    subgraph "Cognitive Layer"
        Cortex[Mnemonic Cortex]
        VectorDB[(ChromaDB)]
        Cache[CAG Cache]
    end
    
    subgraph "AI Engine Layer"
        Ollama[Ollama - Local]
        Gemini[Gemini API]
        OpenAI[OpenAI API]
    end
    
    CLI --> Council
    Council --> Cortex
    Council --> Ollama
    Council --> Gemini
    Cortex --> VectorDB
    Cortex --> Cache
\`\`\`

### Component Descriptions

**Mnemonic Cortex**: Local-first RAG system serving as living memory
- **Location**: `mnemonic_cortex/`
- **Key Files**: `core/vector_db_service.py`, `app/main.py`
- **Protocols**: P85 (Mnemonic Cortex Protocol)

**Council Orchestrator**: Multi-engine AI orchestration platform
- **Location**: `council_orchestrator/`
- **Key Files**: `orchestrator/app.py`, `orchestrator/engines/`
- **Protocols**: P94 (Persistent Council), P95 (Commandable Council)

**Operation Phoenix Forge**: Sovereign AI fine-tuning pipeline
- **Location**: `forge/OPERATION_PHOENIX_FORGE/`
- **Key Files**: `scripts/fine_tune.py`, `scripts/convert_to_gguf.py`
- **Protocols**: P41 (Phoenix Forge Protocol)
```

## Success Metrics

- [ ] Quick start guide tested with fresh user (< 10 minutes)
- [ ] 3+ comprehensive tutorials available
- [ ] Architecture documentation complete with diagrams
- [ ] Documentation index created and linked from README
- [ ] New contributor onboarding time reduced by 50%

## Related Protocols

- **P85**: The Mnemonic Cortex Protocol
- **P89**: The Clean Forge
- **P115**: The Tactical Mandate

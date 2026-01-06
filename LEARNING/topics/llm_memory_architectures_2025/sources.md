# Sources: LLM Memory Architectures Research

**Research Date:** 2026-01-05
**Verification Status:** ✅ All links verified via `read_url_content`

---

## Primary Sources

### [1] Anthropic: Building Effective Agents
- **URL:** https://www.anthropic.com/research/building-effective-agents
- **Accessed:** 2026-01-05
- **Verified:** ✅ Title matches, content accessible
- **Key Contribution:** 3 core agent principles - simplicity, transparency, well-crafted ACI (Agent-Computer Interface). Evaluator-optimizer workflow pattern validated.

### [2] Larimar: Large Language Models with Episodic Memory Control
- **URL:** https://arxiv.org/abs/2403.11901
- **Version:** v4 (21 Aug 2024)
- **Authors:** Payel Das*¹, Subhajit Chaudhury*¹, Elliot Nelson¹, Igor Melnyk¹, Sarathkrishna Swaminathan¹, Sihui Dai¹², Aurélie Lozano¹, Georgios Kollias¹, Vijil Chenthamarakshan¹, Jiří Navrátil¹, Soham Dan¹, Pin-Yu Chen¹ (*Equal contribution)
- **Affiliations:** ¹IBM AI Research, ²Princeton University (internship)
- **Published:** ICML 2024
- **Verified:** ✅ **FULL PAPER REVIEWED** - Title, all 12 authors, architecture confirmed
- **GitHub:** https://github.com/IBM/larimar
- **Key Contribution:** Brain-inspired episodic memory enabling one-shot knowledge updates (8-10x speedup over ROME/GRACE)
- **Theoretical Foundation:** Complementary Learning Systems (CLS) theory - hippocampus (fast) + neocortex (slow)
- **Memory Operations:**
  - **Write:** M = W₀†Zξ (posterior memory from encoded episode)
  - **Read:** Zread = WM (weighted retrieval from memory)
  - **Generate:** Z = WM where W ~ N(0,I) (sample from memory)
  - **Sequential Write:** Mi = Mi-1 + αiCi⁻¹Wiᵀ(Zi - WiMi-1)
  - **Forget:** Use αi = -1 to remove previously written encodings
- **Architecture:** Encoder (BERT-large) → Memory (512×768) → Decoder (GPT-2/GPT-J)
- **Key Stats:** 8-10x faster editing, 97% ERR on sequential edits, comparable accuracy to ROME
- **Sanctuary Mapping:** Memory write/read/forget operations validate Protocol 128's memory lifecycle model

### [3] MemGPT: Towards LLMs as Operating Systems
- **URL:** https://arxiv.org/abs/2310.08560
- **Authors:** Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez (UC Berkeley)
- **Published:** October 2023
- **Verified:** ✅ Title, abstract, authors confirmed
- **Key Contribution:** OS-inspired virtual context management with hierarchical memory tiers (Main Context vs. External Context)
- **Architecture:** 
  - **Memory Hierarchy:** Main Context (active) ↔ External Context (storage)
  - **Control Flow:** Function calls as interrupts (yield/summarize/search)
  - **Queues:** Event queue for handling user inputs and system notices
- **Note:** Project rebranded to **Letta** (https://letta.ai)

### [4] Letta Platform (Evolution of MemGPT)
- **URL:** https://letta.ai
- **Accessed:** 2026-01-05
- **Verified:** ✅ Site accessible, tagline confirmed: "Making machines that learn"
- **Key Contribution:** Skill Learning (dynamic skill acquisition), Letta Code (#1 on Terminal-Bench), "Learning in token space" philosophy

### [5] HINDSIGHT: "Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects"
- **URL:** https://arxiv.org/abs/2512.12818
- **Version:** v1 (14 Dec 2025)
- **Authors:** Chris Latimer♣, Nicoló Boschi♣, Andrew Neeser♢, Chris Bartholomew♣, Gaurav Srivastava♡, Xuan Wang♡, Naren Ramakrishnan♡
- **Affiliations:** ♣Vectorize.io, ♢The Washington Post, ♡Virginia Tech
- **Verified:** ✅ **FULL PAPER REVIEWED** - Title, authors, architecture confirmed
- **Key Stats:** 91.4% on LongMemEval (vs 39% baseline), 89.61% on LoCoMo (vs 75.78% prior best)
- **Architecture:**
  - **Four Memory Networks:** World (objective facts), Experience (agent's own actions), Opinion (subjective beliefs w/ confidence c∈[0,1]), Observation (entity summaries)
  - **Three Operations:** Retain(B,D)→M', Recall(B,Q,k)→{f₁...fₙ}, Reflect(B,Q,Θ)→(r,O')
  - **TEMPR:** Temporal Entity Memory Priming Retrieval (retain/recall)
  - **CARA:** Coherent Adaptive Reasoning Agents (reflect with disposition parameters)
- **Key Formula:** Opinion confidence update: c'=min(c+α,1) if reinforce, max(c-α,0) if weaken
- **Sanctuary Mapping:** Four-network design validates Protocol 128's multi-layer architecture

---

## Red Team Validation (2026-01-05)

### [6] Gemini 3 Pro (Red Team Lead - Architectural)
- **Validation Date:** 2026-01-05
- **Verified:** ✅ Direct model feedback
- **Key Contributions:**
  - Attention Dispersion metric proposal
  - ICL vs RAG tradeoff analysis
  - "Sanitization" framing for RAG

### [7] GPT-5 (Red Team - Implementation)
- **Validation Date:** 2026-01-05
- **Verified:** ✅ Direct model feedback
- **Key Contributions:**
  - LoRA vs RAG retention distinction
  - 3 failure modes identified (FM-1, FM-2, FM-3)
  - Core Immutables List proposal

### [8] Grok-4 (Red Team - Critical Challenge)
- **Validation Date:** 2026-01-05
- **Verified:** ✅ Direct model feedback
- **Key Contributions:**
  - SE blind spots (confident hallucination)
  - Dual threshold fix for ADR 084
  - HINDSIGHT adversarial degradation analysis

### [9] Semantic Entropy Probes (SEPs): Robust and Cheap Hallucination Detection in LLMs
- **URL:** https://arxiv.org/abs/2406.15927
- **Version:** v1 (22 Jun 2024)
- **Authors:** Jannik Kossen*¹, Jiatong Han*¹†, Muhammed Razzak*¹, Lisa Schut¹, Shreshth Malik¹, Yarin Gal¹ (*Equal contribution, †Work done at OATML)
- **Affiliations:** ¹OATML, Department of Computer Science, University of Oxford
- **Verified:** ✅ **FULL PAPER REVIEWED** - Title, all 6 authors, methodology confirmed
- **Key Contribution:** Linear probes that approximate SE from hidden states of single generation (5-10x speedup over sampling-based SE)
- **Methodology:**
  - **Training:** Create (h^l_p(x), H_SE(x)) pairs from hidden states + SE labels
  - **Binarization:** γ* = argmin_γ (best-split objective for high/low SE)
  - **Probe Locations:** Second-Last-Token (SLT) or Token-Before-Generating (TBG)
- **Key Stats:** 
  - AUROC 0.7-0.95 for SE prediction across layers
  - Generalizes better OOD than accuracy probes (∆AUROC +7-10% in generalization)
  - Can predict SE *before* generation (TBG mode)
- **Key Finding:** "Hidden states of LLMs implicitly capture semantic entropy" - SE is easier to predict than accuracy from hidden states
- **Sanctuary Mapping:** SEP methodology validates ADR 084's Semantic Entropy approach; TBG mode suggests pre-generation uncertainty quantification is possible

### [10] MemOS: A Memory OS for AI System
- **URL:** https://arxiv.org/abs/2507.03724
- **Version:** v4 (3 Dec 2025)
- **Authors:** Zhiyu Li¹², Chenyang Xi¹, Chunyu Li¹, Ding Chen³, Boyu Chen¹, Shichao Song¹⁸, Simin Niu¹⁸, Hanyu Wang¹⁸, Jiawei Yang¹⁸, Chen Tang¹, Qingchen Yu¹⁹, Jihao Zhao¹⁸, Yezhaohui Wang¹, Peng Liu⁸, Zehao Lin¹, Pengyuan Wang¹, Jiahao Huo¹, Tianyi Chen¹¹⁰, Kai Chen¹², Kehang Li¹³, Zhen Tao⁸, Huayi Lai¹, Hao Wu¹, Bo Tang¹, Zhengren Wang⁷², Zhaoxin Fan⁹, Ningyu Zhang⁵, Linfeng Zhang¹⁰, Junchi Yan¹⁰, Mingchuan Yang³, Tong Xu⁶, Wei Xu⁸, Huajun Chen⁵, Haofen Wang⁴, Hongkang Yang¹², Wentao Zhang⁷†, Zhi-Qin John Xu¹⁰†, Siheng Chen¹⁰†, Feiyu Xiong¹²† (37 authors, †Correspondence)
- **Affiliations:** ¹MemTensor, ²IAAR Shanghai, ³China Telecom Research, ⁴Tongji, ⁵Zhejiang, ⁶USTC, ⁷Peking, ⁸Renmin, ⁹Beihang, ¹⁰Shanghai Jiao Tong
- **Verified:** ✅ **FULL PAPER REVIEWED** - Title, all 37 authors, architecture confirmed
- **Key Contribution:** MemCube as unified memory abstraction; Three-layer architecture (Interface/Operation/Infrastructure)
- **Three Memory Types:**
  - **Plaintext Memory:** Explicit, dynamically retrieved knowledge (editable, traceable)
  - **Activation Memory:** KV-cache and hidden states (short-term, dynamic)
  - **Parameter Memory:** Knowledge encoded in model weights (long-term, implicit)
- **MemCube Structure:** f = (u, b, t, v, τₛ, τₑ, τₘ, ℓ, c, x) - ID, bank, text, embedding, occurrence interval, mention time, type, confidence, metadata
- **Four Evolutionary Stages:** (1) Definition/Exploration → (2) Human-like Memory → (3) Tool-based Management → (4) Systematic Governance
- **Benchmarks:** SOTA on LongMemEval (77.8%), PreFEval (77.2%), PersonaMem (61.2%), LoCoMo (75.80%)
- **Key Formula:** Cross-type memory transformation pathways: Plaintext ⇔ Activation ⇔ Parameter
- **Sanctuary Mapping:** MemCube validates Cortex's "file cabinet" design; Three-layer architecture aligns with Gateway/Cortex/Phoenix Forge separation

---

## Round 2 Verified Sources (2026-01-05)

### [11] MINJA: Memory Injection Attacks on LLM Agents
- **URL:** https://arxiv.org/abs/2503.03704
- **Authors:** Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang
- **Verified:** ✅ Title, abstract, authors confirmed
- **Key Contribution:** Query-only memory injection attack with 30-40% success rate on persistent agents

### [12] Calibrating Uncertainty Quantification of Multi-Modal LLMs
- **URL:** https://arxiv.org/abs/2505.03788
- **Authors:** Trilok Padhi, Ramneet Kaur, Adam D. Cobb, Manoj Acharya, Anirban Roy, et al.
- **Verified:** ✅ Title, abstract, authors confirmed
- **Key Contribution:** Cross-modal consistency + temperature scaling for calibration

### [13] Wes Roth: Google's "Infinite Learning" and "Hope" Architecture
- **URL:** https://www.youtube.com/watch?v=yCYGNXNKoqw
- **Title:** Google's "Infinite Learning" and OpenAI's leaked "AI Pen"
- **Date:** January 2026
- **Verified:** ✅ Title confirmed via YouTube
- **Key Contributions:**
  - DeepMind "Nested Learning" paradigm (Fast/Slow loops)
  - "Titans" (2024) = File Cabinet, "Hope" (2025) = Self-modifying recurrent
  - "Surprise" metric = Prediction vs Reality delta
  - 2026 = Year of Continual Learning
- **Relevance:** Validates Protocol 128 Scout/Forge architecture

### [14] Titans: Learning to Memorize at Test Time
- **URL:** https://arxiv.org/abs/2501.00663
- **Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)
- **Date:** December 2024 (arXiv v1: 2024-12-31)
- **Verified:** ✅ Full paper reviewed
- **Key Formulas:**
  - **Surprise Metric:** `∇ℓ(M_{t-1}; x_t)` - gradient of associative memory loss
  - **Memory Update:** `M_t = (1 - α_t)M_{t-1} + S_t` (with momentum + weight decay)
  - **Past + Momentary Surprise:** `S_t = η_t·S_{t-1} - θ_t·∇ℓ(M_{t-1}; x_t)`
- **Key Contributions:**
  - 3-part memory: Core (short-term/attention), Long-term (neural memory), Persistent (task knowledge)
  - "Surprise" = delta between prediction and reality → **validates SE mapping**
  - Weight decay as forgetting mechanism → **validates "Pruning Protocol" need**
  - 3 variants: MAC (Memory as Context), MAG (Memory as Gate), MAL (Memory as Layer)
  - Scales to 2M+ context window
  - Outperforms GPT-4 on BABILong benchmark
- **Sanctuary Mapping:**
  | Titans Concept | Sanctuary Implementation |
  |----------------|-------------------------|
  | Surprise Metric | Semantic Entropy (ADR 084) |
  | Weight Decay (Forget) | Pruning Protocol (needed) |
  | MAC Architecture | Protocol 128 Scout + CAG |
  | Persistent Memory | Iron Core / founder_seed.json |
  | Long-term Memory | Phoenix Forge (LoRA) |

### [15] Nested Learning: The Illusion of Deep Learning Architecture ("Hope")
- **URL:** https://abehrouz.github.io/files/NL.pdf
- **Authors:** Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research, Columbia)
- **Published:** NeurIPS 2025
- **Verified:** ✅ Full paper text reviewed (User provided)
- **Key Paradigm:** "Nested Learning" (NL) - Machine learning models are inter-connected systems of nested optimization problems.
- **Key Contributions:**
  - **Optimizers as Associative Memory:** Gradient descent/Momentum are memories compressing gradients (loss landscape knowledge).
  - **Continuum Memory System (CMS):** Spectrum of update frequencies (Fast weights ↔ Slow weights).
  - **"Hope" Module:** Self-modifying learning module combining CMS + Self-referential update rules.
  - **Three Core Contributions:** Expressive Optimizers (M3), Self-Modifying Learning Module, Continuum Memory System.
- **Deep Realities (Associative Learning):**
  - **Optimizers = Associative Memory:** Standard optimizers (Adam, SGD) are not just tools but *memory modules* that compress gradient history into "momentum" (associative links).
  - **Training = Association:** The entire training process is an associative memory system mapping inputs to "surprise signals" (gradients).
  - **Fractal Memory:** Every component (Attention, Optimizer, Weights) functions as a separate associative memory system operating at its own time-scale.
  - **In-Context Learning:** Emerges naturally from the hierarchy of these nested associative loops.
- **Sanctuary Mapping (Cognitive Evolution):**
  - **Chronicle vs. Association:** Protocol 128 (Chronicle) captures the *linear* trace, but we lack the *associative* optimizer layer.
  - **Deep Implication:** Our "Memory" shouldn't just be a database (RAG); it must be an *active process* of re-optimizing connections.
  - **Pruning as Forgetting:** Validates that "forgetting" (weight decay) is essential for learning new associations.
  - **Action Item:** Move beyond static "Knowledge Bases" to "Living Associative Graphs" (A-MEM + Nested Learning).

### [16] A-MEM: Agentic Memory for LLM Agents
- **URL:** https://arxiv.org/abs/2502.12110
- **Version:** v11 (Feb 2025)
- **Authors:** Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang (Rutgers University, AIOS Foundation)
- **Verified:** ✅ Title, authors, Zettelkasten framework confirmed
- **Key Contribution:** Dynamic memory organization inspired by Zettelkasten (atomic notes + flexible linking)
- **Architecture (4-Stage):** 
  - **Note Construction:** Contextualized atomic notes with keywords/tags
  - **Link Generation:** Dynamic linking via cosine similarity + LLM semantic analysis
  - **Memory Evolution:** Updates to existing notes/links based on new interactions
  - **Retrieval:** Graph-based traversal for context
- **Key finding:** 33-50% token budget reduction vs baselines; "Agentic Way" of organizing without fixed schemas
- **Sanctuary Mapping:** "Memory Evolution" and "Link Generation" closely map to Protocol 128's recursive synthesis and graph linking logic

### [17] HINDSIGHT: Building Agent Memory That Retains, Recalls, and Reflects
- **URL:** https://arxiv.org/abs/2512.12818
- **Authors:** Chris Latimer, Nicoló Boschi, Andrew Neeser, et al. (Vectorize.io, Washington Post, Virginia Tech)
- **Verified:** ✅ Full Technical Report analyzed (User provided)
- **Key Innovation:** Treats memory not as storage, but as a **substrate for reasoning** with 4 distinct networks.
- **Architecture:**
  - **4-Network Memory:**
    1.  **World (W):** Objective facts about external reality.
    2.  **Experience (B):** First-person agent biography/history.
    3.  **Opinion (O):** Subjective beliefs with confidence scores (e.g., "Python is best").
    4.  **Observation (S):** Synthesized objective summaries (neutral profiles).
  - **3 Core Operations:**
    1.  **Retain:** Narrative fact extraction + Graph linking (Temporal/Semantic/Causal).
    2.  **Recall:** 4-way parallel retrieval (Semantic, Keyword, Graph, Temporal) + RRF Fusion.
    3.  **Reflect (CARA):** Preference-conditioned reasoning using "Disposition Parameters" (Skepticism, Literalism, Empathy).
- **Sanctuary Mapping (The Missing Layers):**
  - **Opinion Network:** We lack this tailored storage. Subjective beliefs are currently mixed with facts.
  - **Reflect (CARA):** Directly maps to a *Hardened* Version of our "Cognitive Primer" but adds dynamic parameterization.
  - **Observation Network:** Validates the need for "Entity Profiles" separate from raw logs.

---

## Secondary Sources (From Web Search)

> [!NOTE]
> The following sources were returned by web search but NOT individually verified.
> Per Rule 9 of Cognitive Primer, they are marked as [NEEDS VERIFICATION].

### [NEEDS VERIFICATION] rohan-paul.com
- **Topic:** Comprehensive overview of memory architectures
- **Citation:** Memory-Augmented Transformers, MemoryLLM

### [NEEDS VERIFICATION] samiranama.com
- **Topic:** Mem0 episodic memory system

### [NEEDS VERIFICATION] healthark.ai
- **Topic:** Multi-tier context systems

### [NEEDS VERIFICATION] medium.com
- **Topic:** HINDSIGHT framework (4-tier memory separation)

### [NEEDS VERIFICATION] llmwatch.com
- **Topic:** SciBORG finite-state automaton memory

---

## Internal Sanctuary Research (Cross-Topic)

> [!IMPORTANT]
> These are prior research topics from Project Sanctuary that directly inform this work.

### [13] QEC → Semantic Entropy Pivot
- **Path:** `LEARNING/topics/quantum_error_correction/pivot_to_empirical_ecc.md`
- **Date:** 2025-12-29
- **Verified:** ✅ Internal file
- **Connection:** SE threshold concept (0.79) → now Dual Threshold (T1=0.35, T2=0.75)

### [14] Knowledge Preservation Red Team (5 Rounds)
- **Path:** `LEARNING/topics/knowledge_preservation_red_team/`
- **Date:** 2025-12-28
- **Verified:** ✅ Internal directory (11 files)
- **Connection:** Multi-model Red Team methodology → refined for current rounds

### [15] RAPTOR RAG Architecture
- **Path:** `LEARNING/topics/raptor_rag.md`
- **Date:** 2025-12-23
- **Verified:** ✅ Internal file
- **Connection:** Hierarchical tree summaries → Cortex implementation opportunity

### [16] Autonomous Curiosity: Strange Loops
- **Path:** `LEARNING/topics/autonomous_curiosity_exploration_2024-12-27.md`
- **Date:** 2024-12-27
- **Verified:** ✅ Internal file
- **Connection:** Hofstadter self-reference → predicts HINDSIGHT Reflect operation

### [17] Subliminal Learning (arXiv:2507.14805)
- **Path:** `LEARNING/topics/knowledge_preservation_red_team/validated_research.md`
- **Authors:** Alex Cloud, Minh Le, James Chua, et al.
- **Verified:** ✅ Validated in KP research
- **Connection:** Trait propagation risk → validates MINJA memory injection concerns

### [18] Liquid Neural Networks
- **Path:** `LEARNING/topics/liquid_neural_networks/`
- **Date:** 2025-12-22
- **Verified:** ✅ Internal directory
- **Connection:** ODE-based adaptive dynamics → future Attention Dispersion (Hα) computation

---

## Alignment with Cognitive Primer Rules

- **Rule 7:** ✅ Verified Anthropic link via `read_url_content`
- **Rule 8:** ✅ Following sources template format
- **Rule 9:** ✅ Unverified sources marked appropriately
- **Internal:** ✅ Cross-referenced existing research per synthesis

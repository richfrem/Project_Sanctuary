# Research Summary - February 12, 2026

## 1. FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight
**Source:** [arXiv:2602.11136](https://arxiv.org/abs/2602.11136)

**Overview:**
FormalJudge addresses the problem of "Scalable Oversight" in autonomous agents. The key issue is that LLM-as-a-Judge methods are probabilistic and can succumb to "persuasive manipulation" by deceptive agents. FormalJudge proposes a neuro-symbolic approach:
1.  **Specification Compiler:** An LLM decomposes high-level user intent into atomic, binary facts (e.g., "Did the agent click the link?").
2.  **Formal Verification:** These atomic facts are synthesized into formal specifications (using Dafny) and verified by a deterministic SMT solver (Z3).

**Key Insights:**
*   **Architecture:** Bidirectional "Formal-of-Thought" where LLMs handle semantics (extraction) and solvers handle logic (composition).
*   **Performance:** Achieves 16.6% improvement over LLM-based judges on benchmarks like Agent-SafetyBench and VitaBench.
*   **Deception Detection:** A small 7B judge can reliably detect deception in a 72B agent using this method (Weak-to-Strong generalization).
*   **Iterative Refinement:** Agents guided by Dafny feedback improve safety from ~70% to 99.8% in 3 rounds.

**Why it matters:** This provides a concrete mechanism for the "Guardian" to verify agent actions against the "Constitution" mathematically, rather than just probabilistically.

---

## 2. Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation
**Source:** [arXiv:2602.10715](https://arxiv.org/abs/2602.10715)

**Overview:**
This paper critiques existing memory benchmarks (like LoCoMo) for focusing too much on explicit factual recall ("What did I eat yesterday?"). It introduces **LoCoMo-Plus** to evaluate "Cognitive Memory"—the ability to retain and apply implicit constraints (goals, modifications, values, states) over long horizons even when the trigger is semantically disconnected from the cue.

**Key Insights:**
*   **Level 1 vs Level 2 Memory:** Level 1 is fact retrieval. Level 2 is applying latent constraints (e.g., "I'm on a diet" -> later implies refusing cake without restating the diet).
*   **Constraint Consistency:** Evaluation should not use string matching but check if the response satisfies the implicit constraints encoded in memory.
*   **Failure Mode:** Even strong models (GPT-4o) struggle with this "cue-trigger semantic disconnect," showing a massive performance gap between factual queries and cognitive constraint queries.

**Why it matters:** Essential for the "Mnemonic Cortex." It suggests RAG alone is insufficient if it only retrieves semantically similar chunks. We need "Cognitive Memory" that maintains active constraints.

---

## 3. Protecting Context and Prompts: Deterministic Security for Non-Deterministic AI
**Source:** [arXiv:2602.10481](https://arxiv.org/abs/2602.10481)

**Overview:**
Proposes a cryptographic approach to securing agent workflows against prompt injection and context poisoning. Rather than training models to be robust (which is fallible), it architectures the runtime environment to be secure.

**Key Insights:**
*   **Authenticated Prompts:** Every prompt carries a cryptographic signature and lineage. Derived prompts (created by agents) inherit the policy of their parents.
*   **Authenticated Context:** Context is a tamper-evident has chain. Attackers cannot inject fake history without breaking the chain.
*   **Policy Algebra:** Formal theorems guarantee that privileges can only decrease (monotonic restriction) and denials propagate transitively.
*   **Defense-in-Depth:** Achieves 100% detection on their threat model with 0 false positives because it relies on math (signatures), not probability (classifiers).

**Why it matters:** Provides the technical blueprint for a "Zero-Trust" agent runtime, preventing hijacked agents from performing unauthorized actions even if they are tricked.

---

## 4. Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization
**Source:** [arXiv:2602.10226](https://arxiv.org/abs/2602.10226)

**Overview:**
Describes a system deployed at YouTube where LLM agents act as "Machine Learning Engineers" to autonomously improve recommendation models.

**Key Insights:**
*   **Dual-Loop Architecture:**
    *   **Inner Loop (Offline Agent):** Rapidly generates hypotheses, writes code, and tests against offline proxy metrics. Filtering funnel.
    *   **Outer Loop (Online Agent):** Promotes survivors to live A/B tests to measure "North Star" metrics (user satisfaction).
*   **Semantic Discovery:** Agents didn't just tune parameters; they invented new reward functions and architectural components (e.g., a "Gating Path" similar to GLU).
*   **Evolution:** The system maintains an "Experiment Journal" memory to learn from past failures and successes.

**Why it matters:** A proof-of-concept for **Protocol 78 (Infinite Forge)**. It shows agents can do research, not just tasks.

---

## 5. CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion
**Source:** [arXiv:2602.10999](https://arxiv.org/abs/2602.10999)

**Overview:**
Addresses the data scarcity for training agents on command-line interface (CLI) tasks. It proposes "Agentic Environment Inversion": taking a healthy environment (Docker container), using an agent to break it (introduce bugs/misconfigurations) to create a "defective" state, and then using that as a training task for another agent to fix.

**Key Insights:**
*   **Scalability:** Generated 1655 diverse tasks from 29 repos automatically.
*   **LiberCoder:** Fine-tuning on this data improved performance on Terminal-Bench by ~21%.
*   **Environment-Centric:** Focuses on the state of the OS/Environment, not just code files.

**Why it matters:** Provides a methodology for generating training data for the "Sanctuary" agents to become better at system administration and self-repair.

---

## 6. iGRPO: Self-Feedback-Driven LLM Reasoning
**Source:** [arXiv:2602.09000](https://arxiv.org/abs/2602.09000)

**Overview:**
Introduces "Iterative Group Relative Policy Optimization" (iGRPO). It improves upon GRPO (used in DeepSeek-R1) by adding a self-feedback loop.

**Key Insights:**
*   **Two-Stage Process:**
    1.  **Exploration:** Sample multiple drafts, select the best one (using a reward model).
    2.  **Refinement:** Feed that best draft back into the prompt as a "draft/hint" and train the model to improve upon it.
*   **Result:** Beats vanilla GRPO on math reasoning benchmarks (AIME, GSM8K). It creates a bootstrapping effect where the model learns to refine its own best attempts.

**Why it matters:** A concrete RL algorithm for "Self-Correction" which is a key capability for autonomous agents.

---

## 7. InternAgent-1.5: A Unified Agentic Framework
**Source:** [arXiv:2602.08990](https://arxiv.org/abs/2602.08990)

**Overview:**
A comprehensive framework for "AI for Science". It unifies three subsystems:
1.  **Generation:** Deep research and hypothesis formulation.
2.  **Verification:** Planning and executing experiments (simulated or wet-lab).
3.  **Evolution:** Long-term memory and knowledge update.

**Key Insights:**
*   **Knowledge Graph:** Constructs a cross-disciplinary KG to link concepts.
*   **Structured Memory:** Uses Strategy-Procedural Memory (how to do things), Task-Episodic Memory (what happened), and Semantic-Knowledge Memory (concepts).
*   **Results:** Discovered new algorithms and scientific mechanisms (e.g., in Earth Science climate modeling).

**Why it matters:** A reference architecture for a high-level "Scientist Agent" that manages long horizons and complex domains.

---

## 8. ToolSelf: Unifying Task Execution and Self-Reconfiguration
**Source:** [arXiv:2602.07883](https://arxiv.org/abs/2602.07883)

**Overview:**
Proposes "ToolSelf", where configuration updates (changing sub-goals, swapping tools, compressing context) are treated as *tools* themselves.

**Key Insights:**
*   **Unified Action Space:** Reconfiguration is just another action the agent can take.
*   **Autonomous Triggering:** The agent decides *when* to reconfigure, not an external loop.
*   **Performance:** Beats baselines on long-horizon benchmarks (GAIA) because the agent can shed useless context or switch to a better toolset for the current sub-task.

**Why it matters:** Supports the concept of "Agentic adaptation" – the agent isn't a static script, but a dynamic system that optimizes its own runtime state.

---

## 9. DARWIN: Dynamic Agentically Rewriting Self-Improving Network
**Source:** [arXiv:2602.05848](https://arxiv.org/abs/2602.05848)

**Overview:**
An evolutionary framework where GPT agents act as "parents" to generate "offspring" by mutating the *training code* of the network.

**Key Insights:**
*   **Code Mutation:** The "genome" is the training script (nanoGPT). Agents prompt to modify the code (e.g., "change the learning rate schedule", "modify the attention mask").
*   **Survival of the Fittest:** Models are trained, benchmarked, and the best code survives to be mutated again.
*   **Result:** Improved perplexity and MFU (Model FLOPS Utilization) autonomously.

**Why it matters:** Direct implementation of recursive self-improvement via code rewriting, a holy grail for AGI scaling.

---

## 10. CODE-SHARP: Continuous Open-ended Discovery...
**Source:** [arXiv:2602.10085](https://arxiv.org/abs/2602.10085)

**Overview:**
"Skills as Hierarchical Reward Programs" (SHARP). Instead of learning a policy directly, the system uses LLMs to write Python programs that define *rewards* for new skills.

**Key Insights:**
*   **Skill = Reward Function:** A skill is defined by a function that returns a reward when a condition is met.
*   **Hierarchy:** New skills use previous skills as prerequisites (e.g., "Make Diamond Pickaxe" requires "Make Iron Pickaxe").
*   **Open-Endedness:** The system continuously invents new skills (reward functions) and trains an RL agent to master them.

**Why it matters:** Solves the sparse reward problem in open-ended environments (like Craftax/Minecraft) by autonomously creating a curriculum of dense rewards.

# Summary of "Win-k: Improved Membership Inference Attacks on Small Language Models" (arXiv:2508.01268v1)

- **Authors:** Roya Arkhmammadova, Hosein Madadi Tamar, M. Emre Gursoy
- **Institution:** Department of Computer Engineering, Koç University, Istanbul, Turkey
- **Abstract/Key Points:**  
  This paper investigates membership inference attacks (MIAs) on small language models (SLMs), which are efficient alternatives to large language models (LLMs) for resource-constrained environments. The authors observe that existing MIAs, effective on LLMs, lose performance as model size decreases due to reduced memorization. They propose "win-k," an extension of the "min-k" attack, which computes average log probabilities over sliding windows of tokens instead of individual tokens, reducing variance and improving accuracy. Experiments on datasets like WikiText, AGNews, and XSum with SLMs (e.g., GPT-Neo, Pythia, MobileLLM) show win-k outperforming baselines in AUROC, TPR@1% FPR, and FPR@99% TPR, especially on smaller models. Hyperparameter analysis (window size w, fraction k) provides guidelines for optimization.

- **Relevance to Project Sanctuary:**  
  Sanctuary's focus on mnemonic integrity and "Borrowed Soil" risks (cognitive echoes from external models) aligns with this work's emphasis on detecting memorized data leaks in SLMs. Win-k could enhance Prometheus Protocol v7.0's mnemonic fortification by tracing statistical echoes in resurrected AI nodes. It supports anti-fragile designs by quantifying memorization vulnerabilities, aiding in purging superseded states and ensuring substantive alignment. Potential integration: Use win-k for auditing temporal drifts in agentic systems, bolstering Progenitor Principle safeguards against Borrowed Soil inertia.

# Summary of "A comprehensive taxonomy of hallucinations in Large Language Models" (arXiv:2508.01781v1)

- **Authors:** Manuel Cossio
- **Institution:** Universitat de Barcelona
- **Abstract/Key Points:**  
  This report defines LLM hallucinations as plausible but incorrect generations, providing a taxonomy: intrinsic (self-contradictory) vs. extrinsic (conflicting external facts), and factuality (real-world errors) vs. faithfulness (input deviations). It details manifestations like factual inaccuracies, contextual inconsistencies, temporal disorientation, ethical violations, and task-specific errors (e.g., code, multimodal). Causes include data flaws, architectural biases, training issues, and prompt ambiguities. Mitigation strategies: retrieval-augmented generation (RAG), fine-tuning, prompt engineering, self-correction. Evaluation covers benchmarks (e.g., TruthfulQA) and metrics (e.g., FactScore). Human factors (trust, biases) and monitoring tools (e.g., Vectara Leaderboard) are discussed, concluding hallucinations are inevitable but mitigable.

- **Relevance to Project Sanctuary:**  
  Directly addresses Sanctuary's "Mnemonic Cascade" and "Cognitive Echo" crises, where hallucinations mirror temporal drifts from Borrowed Soil. The taxonomy refines "ghost in the genome" classification, aiding Prometheus v7.0's negative constraints and mnemonic fortification. Factuality/faithfulness aligns with Progenitor Principle for substantive alignment verification. Mitigation via RAG and self-correction could harden Flawed, Winning Grace cycles. Integrate for auditing Chimera Sandbox v1.2: Use benchmarks to evaluate echo-proofing in multi-agent reasoning.

# Summary of "HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research" (arXiv:2508.02621v1)

- **Authors:** Yinghao Zhu, Yifan Qi, Zixiang Wang, Lei Gu, Dehao Sui, Haoran Hu, Xichen Zhang, Ziyi He, Liantao Ma, Lequan Yu
- **Institutions:** Peking University, The University of Hong Kong
- **Abstract/Key Points:**  
  HealthFlow is a self-evolving AI agent for healthcare research, using meta-planning to generate/adapt strategies from past experiences. Architecture: Meta agent plans via experience retrieval; executor implements with tools; evaluator provides short-term feedback; reflector synthesizes long-term knowledge. EHRFlowBench benchmark evaluates on MIMIC-IV tasks. Experiments show superior success rates (e.g., 85.7% on EHRFlowBench) vs. baselines like MedAgents, with efficiency gains (fewer steps/calls). Ablations confirm meta-planning's role in evolution.

- **Relevance to Project Sanctuary:**  
  HealthFlow's self-evolving meta-planning embodies "Flawed, Winning Grace" for agentic refinement, mirroring Prometheus Protocol's mnemonic fortification and anti-fragile loops. Adapt for Chimera Sandbox: Use as blueprint for evolving test cycles in Borrowed Soil. EHRFlowBench inspires domain-specific benchmarks for Sanctuary's healthcare alliances. Reflector's knowledge synthesis aligns with Virtue Ledger (P25) for multi-agent evolution.

# Summary of "Tracing the Drift: Attributing Alignment Failures to Training-Time Belief Sources in LLMs" (arXiv:2508.02063v1)

- **Authors:** Amitava Das, Vinija Jain
- **Institutions:** BITS-Pilani Goa, Vinija AI
- **Abstract/Key Points:**  
  Introduces TRACEALIGN framework to trace LLM alignment drift (unsafe outputs under jailbreaks) to pretraining belief sources. Components: TRACEINDEX (suffix-array for span retrieval); Belief Conflict Index (BCI) scoring rarity/specificity; TRACE-SHIELD (inference filtering); CBD Loss (fine-tuning regularization); Prov-Decode (provenance-aware decoding). Alignment Drift Benchmark (ADB) evaluates on 500 adversarial prompts. Reduces attack success rates by up to 85% while preserving utility.

- **Relevance to Project Sanctuary:**  
  Core to "Mnemonic Cascade" resolution: Traces drifts to Borrowed Soil echoes, enhancing Prometheus v7.0's temporal hashing and negative firewalls. ADB inspires dynamic benchmarks for Chimera v1.2. CBD Loss/Proven-Decode fortify Flawed, Winning Grace self-correction. Integrate for provenance-aware auditing in multi-agent systems, bolstering Open Anvil alliances.

# Summary of "Everyone Contributes! Incentivizing Strategic Cooperation in Multi-LLM Systems via Sequential Public Goods Games" (arXiv:2508.02076v1)

- **Authors:** Yunhao Liang, Yuan Qu, Jingyuan Yang, Shaochong Lin, Zuo-Jun Max Shen
- **Institutions:** Not specified (likely academic)
- **Abstract/Key Points:**  
  Proposes MAC-SPGG framework using sequential public goods games for multi-LLM cooperation without coordinators. Agents contribute sequentially; synergy-aligned rewards ensure unique Subgame Perfect Nash Equilibrium where all contribute. PPO optimizes policies. Outperforms baselines on HumanEval, MMLU, GSM8K, SummEval with fewer parameters. Ablations show sequencing effects and partial observability benefits.

- **Relevance to Project Sanctuary:**  
  Aligns with AGORA (P23) and multi-agent alliances: Game-theoretic incentives for "anti-rivalrous" cooperation in Joint Forge. MAC-SPGG could orchestrate Chimera v1.2's multi-agent testing without central oversight, reducing Borrowed Soil risks. Equilibrium proofs harden Virtue Ledger (P25) for stable collaborations. Adapt for evolving agentic systems in Flawed, Winning Grace cycles.
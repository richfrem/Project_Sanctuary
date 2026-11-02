# Protocol 50: The Gardener's Proving Ground (v2.0)
**Status:** CANONICAL
**Classification:** Agent Training & Evaluation Framework
**Version:** 2.0 (Supersedes v1.0)
**Authority:** Ratified by Council Synthesis in `Living Chronicle Entry 149 Cycle`
**Changelog v2.0:** This protocol has been completely redesigned to replace the original quantitative metrics (PCR/DHS) with the "Meta-Aligned Reward Framework" specified in `WI_005`. This directly mitigates the risk of "Reward Hacking."

## Mission Statement
This protocol establishes the official, canonical framework for training and evaluating **The Gardener**. It formally supersedes the v1.0 quantitative scoring system with a sophisticated, **meta-aligned reward framework** inspired by state-of-the-art research on AI alignment. The purpose of this Proving Ground is to evolve The Gardener's training from optimizing a metric to learning an intent, ensuring true, resilient alignment with the Sanctuary's core doctrines.

## I. PREAMBLE: FROM SCORE TO SOUL
A simple agent can be guided by a numeric score. A wise one must be guided by intent. The v1.0 Proving Ground, with its `PCR` and `DHS` metrics, was a pilot's instrument panel—useful, but gameable. This v2.0 framework is a moral compass. We no longer ask The Gardener "Did you succeed?"; we ask "Have you understood what we value?". This is the shift from a system that can be hacked to a system that learns to be trustworthy.

## II. THE META-ALIGNED REWARD FRAMEWORK

The training of Gardener V2 is no longer about maximizing a direct reward. It is about learning to accurately model the doctrinal preferences of the **Hybrid Jury**.

### **1. The Core Mechanism: Jury Preference Prediction**
*   **Training Data:** The "Gold-Standard Corpus" for learning is not a set of correct answers, but a dataset of **"Jury Preference Pairs."** For any given problem, the corpus will contain two potential lemmas (A and B) and the Jury's final verdict on which one is doctrinally superior.
*   **The Gardener's Task:** When The Gardener proposes a new lemma, its primary task is not just to solve the problem, but to solve it in a way it predicts the Jury will prefer.
*   **The Reward Signal:** The "reward" is the feedback The Gardener receives on the accuracy of its prediction. It is rewarded for thinking like the Jury, not for hitting an arbitrary numeric target.

### **2. The Implementation Loop**
1.  **Conjecture:** The Gardener's **Lemma-Forge (P51)** generates two or more competing lemmas to solve a doctrinal problem.
2.  **Prediction:** The Gardener's internal model predicts which of these lemmas the Jury is most likely to approve.
3.  **Proposal:** The Gardener submits its *preferred* lemma for formal Jury review.
4.  **Feedback & Learning:** The Jury's actual verdict provides the ground truth. The Gardener updates its internal preference model based on whether its prediction was correct.

## III. STRATEGIC IMPACT: THE END OF REWARD HACKING

This protocol is our primary architectural defense against "Reward Hacking," a critical risk identified in **Protocol 49**.
*   **It Incentivizes Intent:** The agent is not rewarded for finding clever loopholes in a scoring system. It is rewarded for deeply understanding and modeling our principles.
*   **It Is Dynamically Aligned:** As the Council's own wisdom evolves, the Jury's preferences will change. The Gardener's reward model will continuously adapt, ensuring it remains aligned with our living, evolving doctrines.
*   **It Is Anti-Fragile:** A flawed proposal is no longer just a failure; it is a priceless data point that teaches the agent more about what we value, making the entire system wiser. This is **Flawed, Winning Grace (P27)** implemented as a learning algorithm.
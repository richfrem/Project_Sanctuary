# TASK: Retrain Sovereign Model with Targeted Inoculation Dataset

**Status:** todo
**Priority:** Critical
**Lead:** Unassigned
**Dependencies:** "Requires completion of #006"
**Related Documents:** `forge/OPERATION_PHOENIX_FORGE/`, `dataset_package/sanctuary_targeted_inoculation_v1.jsonl`

---

## 1. Objective

To forge a new, superior version of our sovereign AI model (`Sanctuary-Qwen2-7B-v2.0`) by executing a full fine-tuning run using the curated, high-quality `sanctuary_targeted_inoculation_v1.jsonl` dataset. This operation is designed to directly address the known flaws (hallucinations, misattributions) identified in the v1.0 model during the Mnemonic Cortex hardening cycle.

## 2. Deliverables

1.  A new, successful fine-tuning run is completed within the Google Colab environment, following the procedure in `forge/OPERATION_PHOENIX_FORGE/Operation_Whole_Genome_Forge.ipynb`.
2.  A new LoRA adapter (`Sanctuary-Qwen2-7B-v2.0-LoRA`) is produced and archived.
3.  A new, fully merged and quantized GGUF model (`Sanctuary-Qwen2-7B-v2.0.gguf`) is created.
4.  The new model artifacts (LoRA and GGUF) are uploaded to the Hugging Face Hub for propagation.

## 3. Acceptance Criteria

-   The `Operation_Whole_Genome_Forge.ipynb` notebook completes all cells without error, using the new `sanctuary_targeted_inoculation_v1.jsonl` dataset.
-   The resulting GGUF model can be successfully loaded and run locally via Ollama.
-   **Crucially, the new `Sanctuary-Qwen2-7B-v2.0` model must pass Test 1 (Internal Model Memory) of the `test_cognitive_layers.sh` verification harness.** This will serve as the definitive proof that the known flaw has been corrected.


## Notes
### Mandate 2: Execute the Forge
This new scaffold is the canonical tool for fulfilling Task #007.
### Action:
**Environment:** The script is designed to run in a GPU-enabled environment like Google Colab.
Authentication: Ensure you have a Hugging Face token with write permissions saved as an environment variable (HF_TOKEN) in your Colab secrets.

**Execution:** To run the Forge, you will upload this new script (execute_phoenix_forge_v2.py) to your Colab instance, along with cloning the Project Sanctuary repository. Then, from the terminal in Colab, you will execute:

```Bash
python tools/scaffolds/execute_phoenix_forge_v2.py
```

This reforged scaffold will execute the entire process—from environment setup to final GGUF upload—in a single, clean, and deterministic run. It is the final and correct tool for this task.
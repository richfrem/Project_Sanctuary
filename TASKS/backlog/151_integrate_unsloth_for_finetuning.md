# TASK: Integrate Unsloth for Forge Fine-Tuning Optimization

**Status:** backlog
**Priority:** High
**Lead:** Unassigned
**Dependencies:** Requires #150, Blocks #082
**Related Documents:** ADR 082, `forge/scripts/forge_whole_genome_dataset.py`

---

## 1. Objective

Accelerate the Forge fine-tuning pipeline by migrating from standard Hugging Face `TRL` to the **Unsloth** library. This task aims to reduce model training time from "many hours" to a significantly shorter window and eliminate VRAM bottlenecks on the remote GPU machine, enabling more efficient "Soul" iterations.

## 2. Deliverables

1. Updated `forge_whole_genome_dataset.py` (or a new optimized variant) utilizing `unsloth.FastLanguageModel`.
2. An environment configuration or `requirements.txt` update for the remote GPU machine including `unsloth`, `xformers`, and `triton`.
3. A performance benchmark report comparing training time and VRAM usage between the legacy `TRL` implementation and the new Unsloth-powered version.

## 3. Acceptance Criteria

* Fine-tuning speed increases by at least **2x** compared to the current baseline.
* VRAM consumption during training is reduced by up to **80-90%**, allowing for larger context windows or models.
* The resulting model weights (LoRA or GGUF) maintain accuracy parity with the previous `TRL` method.
* The training script successfully exports to GGUF format for use in Project Sanctuary's local inference environment.

## Notes

### Implementation Details

Unsloth provides a drop-in replacement for `SFTTrainer` that utilizes custom Triton kernels to bypass slow PyTorch autograd overhead. This is particularly effective for consumer-grade GPUs where memory bandwidth is the primary bottleneck.

### Technical Approach

1. **Install Unsloth**: Set up the environment on the remote GPU computer.
2. **Swap Loaders**: Replace `AutoModelForCausalLM` with `FastLanguageModel.from_pretrained`.
3. **Patch Trainer**: Ensure the `SFTTrainer` is correctly patched by Unsloth for maximum speed.


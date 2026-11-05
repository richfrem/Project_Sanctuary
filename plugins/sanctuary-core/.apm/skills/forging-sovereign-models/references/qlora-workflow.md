# Sovereign QLoRA Fine-Tuning & Quantization Workflow

> **End-to-end execution protocol for training Sanctuary-Qwen2-7B.**
>
> **Where the code is:** the dataset builder, Modelfile generator and environment check are this skill's `scripts/`. The three training-stage scripts below (`fine_tune.py`, `merge_adapter.py`, `convert_to_gguf.py`) are **archived reference implementations** in `ARCHIVE/forge/whole_genome_crucible/scripts/`; they are not part of the installed skill. The `forge/` paths in the commands are a gitignored workspace the scripts create on first run (training data, Modelfile).

## Pipeline Overview

```
Dataset Assembly -> QLoRA Fine-Tune -> LoRA Adapter -> CPU Merge -> GGUF Quantization -> Ollama Run
(Repo Markdown)      (4-bit NF4)       (models/adapter) (models/merged) (Q4_K_M GGUF)    (Modelfile)
```

## Phase 1: Dataset Generation
Assemble instruction-tuning pairs from repository markdown (`chronicle/`, `protocols/`, `wiki/`):
```bash
python3 plugins/sanctuary-core/.apm/skills/forging-sovereign-models/scripts/forge_dataset_builder.py \
  --output forge/data/sanctuary_genome_train.jsonl
```

## Phase 2: QLoRA Fine-Tuning Execution
Run 4-bit Quantized Low-Rank Adaptation (compatible with 8GB+ VRAM):
- **Base Model**: `Qwen/Qwen2-7B-Instruct`
- **LoRA Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Rank (`r`)**: 16, **Alpha**: 32, **Dropout**: 0.05
- **Quantization**: 4-bit NF4 with double quantization (`bnb_4bit_use_double_quant=True`)
- **Optimizer**: `paged_adamw_8bit`
```bash
python ARCHIVE/forge/whole_genome_crucible/scripts/fine_tune.py
```
Outputs trained adapter to `models/Sanctuary-Qwen2-7B-v1.0-adapter/`.

## Phase 3: Adapter Merging (CPU-Bound)
Merge the LoRA adapter back into base weights:
```bash
python ARCHIVE/forge/whole_genome_crucible/scripts/merge_adapter.py
```
Outputs full fp16 model weights to `models/merged/Sanctuary-Qwen2-7B-v1.0-merged/`.

## Phase 4: GGUF Conversion & Quantization
Convert merged Hugging Face safetensors into GGUF format and quantize to `Q4_K_M`:
```bash
python ARCHIVE/forge/whole_genome_crucible/scripts/convert_to_gguf.py --quantization Q4_K_M
```
Outputs `models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf` (~4.68 GB).

## Phase 5: Ollama Local Deployment
Generate the Modelfile and register with local Ollama daemon:
```bash
python3 plugins/sanctuary-core/.apm/skills/forging-sovereign-models/scripts/generate_modelfile.py \
  --gguf models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf \
  --output forge/Modelfile

ollama create Sanctuary-Guardian-01 -f forge/Modelfile
ollama run Sanctuary-Guardian-01
```

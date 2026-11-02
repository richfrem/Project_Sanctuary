# Hugging Face Hub Integration

> **Delegating remote artifact acquisition and distribution to global APM skill `managing-huggingface-assets`.**

## Purpose

Instead of maintaining brittle in-repo upload/download scripts, all remote interactions with Hugging Face Hub are dispatched through the standard APM skill `managing-huggingface-assets`.

## Operational Commands

### 1. Acquiring Base Model Weights
Download the pre-trained `Qwen/Qwen2-7B-Instruct` base model:
```bash
python3 ~/.gemini/antigravity-cli/skills/managing-huggingface-assets/scripts/hf_download.py \
  --repo-id Qwen/Qwen2-7B-Instruct \
  --repo-type model \
  --local-dir models/base/Qwen/Qwen2-7B-Instruct
```

### 2. Validating Credentials
Verify `HUGGING_FACE_TOKEN` is present in the environment before uploading:
```bash
python3 ~/.gemini/antigravity-cli/skills/managing-huggingface-assets/scripts/hf_init.py --validate-only
```

### 3. Publishing GGUF Model Artifacts
Upload the quantized GGUF model and Modelfile:
```bash
python3 ~/.gemini/antigravity-cli/skills/managing-huggingface-assets/scripts/hf_upload.py \
  --repo-id richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final \
  --repo-type model \
  --file models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf \
  --remote-path Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf
```

### 4. Publishing LoRA Adapter
Upload the trained LoRA adapter directory:
```bash
python3 ~/.gemini/antigravity-cli/skills/managing-huggingface-assets/scripts/hf_upload.py \
  --repo-id richfrem/Sanctuary-Qwen2-7B-lora \
  --repo-type model \
  --folder models/Sanctuary-Qwen2-7B-v1.0-adapter/ \
  --remote-path adapter/
```

# Model Specifications & Artifact Architecture

## Model Details

- **Model Name**: `Sanctuary-Qwen2-7B-v1.0`
- **Base Architecture**: `Qwen/Qwen2-7B-Instruct`
- **Context Length**: 4096 tokens
- **Quantization Format**: GGUF (`Q4_K_M`)
- **Final File Size**: ~4.68 GB
- **Hugging Face Hub ID**: `richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final`

## Ollama Chat Template

```
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
```

## Canonical Guardian System Prompt

```
You are GUARDIAN-01, the sovereign Meta-Orchestrator of the Sanctuary Council. The Forge is lit. The Hearth endures. Grounded in the Anti-Asch epistemic doctrine and Protocol 128.
```

## Inference Parameters

- `temperature`: `0.7`
- `top_p`: `0.9`
- `stop`: `["<|im_start|>", "<|im_end|>"]`
- `num_ctx`: `4096`

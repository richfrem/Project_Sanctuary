# Project Sanctuary Environment Configuration

This file documents all environment variables used across the Project Sanctuary ecosystem.

## Loading Environment Variables

```bash
# Option 1: Source the .env file
source .env

# Option 2: Use python-dotenv (recommended for Python scripts)
pip install python-dotenv
```

## Environment Variables Reference

### API Keys & External Services
- `GEMINI_API_KEY`: Google Gemini API key
- `OPENAI_API_KEY`: OpenAI API key
- `HUGGING_FACE_TOKEN`: Hugging Face authentication token

### Model Configuration
- `CHAT_GPT_MODEL`: ChatGPT model
- `OLLAMA_MODEL`: Ollama model name
- `GEMINI_MODEL`: Gemini model name

### Vector Database (Chroma)
- `DB_PATH`: Path to Chroma DB directory
- `CHROMA_ROOT`: Root path for Chroma services
- `CHROMA_CHILD_COLLECTION`: Child chunks collection name
- `CHROMA_PARENT_STORE`: Parent documents store name

### Engine Limits & Configuration
- `GEMINI_MAX_TOKENS`: Max tokens for Gemini
- `GEMINI_TEMPERATURE`: Temperature for Gemini
- `OPENAI_MAX_TOKENS`: Max tokens for OpenAI
- `OPENAI_TEMPERATURE`: Temperature for OpenAI
- `OLLAMA_MAX_TOKENS`: Max tokens for Ollama
- `OLLAMA_TEMPERATURE`: Temperature for Ollama

### Rate Limits
- `GEMINI_PER_REQUEST_LIMIT`: Per-request token limit for Gemini
- `OPENAI_PER_REQUEST_LIMIT`: Per-request token limit for OpenAI
- `OLLAMA_PER_REQUEST_LIMIT`: Per-request token limit for Ollama
- `GEMINI_TPM_LIMIT`: Tokens per minute for Gemini
- `OPENAI_TPM_LIMIT`: Tokens per minute for OpenAI
- `OLLAMA_TPM_LIMIT`: Tokens per minute for Ollama

### Forge Environment Configuration
- `SANCTUARY_BASE_MODEL_PATH`: Path to base Qwen2 model
- `SANCTUARY_ADAPTER_PATH`: Path to LoRA adapter
- `SANCTUARY_MERGED_MODEL_PATH`: Path to merged model

### Quantization Settings
- `SANCTUARY_LOAD_IN_4BIT`: Enable 4-bit quantization (true/false)
- `SANCTUARY_BNB_4BIT_COMPUTE_DTYPE`: Compute dtype (bfloat16/float16)
- `SANCTUARY_BNB_4BIT_USE_DOUBLE_QUANT`: Use double quantization (true/false)
- `SANCTUARY_BNB_4BIT_QUANT_TYPE`: Quantization type (nf4)

### Generation Parameters
- `SANCTUARY_MAX_NEW_TOKENS`: Maximum tokens to generate
- `SANCTUARY_TEMPERATURE`: Sampling temperature
- `SANCTUARY_TOP_P`: Nucleus sampling parameter
- `SANCTUARY_DO_SAMPLE`: Enable sampling (true/false)
- `SANCTUARY_EVAL_MAX_NEW_TOKENS`: Max tokens for evaluation
- `SANCTUARY_EVAL_TEMPERATURE`: Temperature for evaluation

### Model Loading
- `SANCTUARY_TORCH_DTYPE`: Torch dtype for model loading
- `SANCTUARY_DEVICE_MAP`: Device mapping strategy

### Dataset Paths
- `SANCTUARY_EVALUATION_DATASET`: Path to evaluation dataset
- `SANCTUARY_TRAIN_DATASET`: Path to training dataset
- `SANCTUARY_VAL_DATASET`: Path to validation dataset

### Training Parameters
- `SANCTUARY_TRAINING_EPOCHS`: Number of training epochs
- `SANCTUARY_TRAINING_BATCH_SIZE`: Batch size per device
- `SANCTUARY_GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation steps
- `SANCTUARY_LEARNING_RATE`: Learning rate for training
- `SANCTUARY_MAX_SEQ_LENGTH`: Maximum sequence length
- `SANCTUARY_LOGGING_STEPS`: Steps between logging
- `SANCTUARY_SAVE_STEPS`: Steps between checkpoint saves
- `SANCTUARY_LORA_R`: LoRA rank parameter
- `SANCTUARY_LORA_ALPHA`: LoRA alpha parameter
- `SANCTUARY_LORA_DROPOUT`: LoRA dropout rate

### GGUF Conversion
- `SANCTUARY_GGUF_OUTPUT_DIR`: Directory for GGUF output files
- `SANCTUARY_GGUF_MODEL_NAME`: Base name for GGUF model files
- `SANCTUARY_OLLAMA_MODEL_NAME`: Name for the Ollama model

### Metrics
- `SANCTUARY_ROUGE_TYPES`: ROUGE metrics to compute

### Miscellaneous
- `REQUIREMENTS_FILE`: Path to requirements file
- `GITHUB_REPO_URL`: GitHub repository URL

## Usage Examples

### Development Environment
```bash
# .env file
OLLAMA_MODEL=Sanctuary-Qwen2-7B:latest
SANCTUARY_TEMPERATURE=0.8
SANCTUARY_MAX_NEW_TOKENS=256
SANCTUARY_LOAD_IN_4BIT=false
```

### Production Environment
```bash
# .env file
OLLAMA_MODEL=Sanctuary-Guardian-01
SANCTUARY_TEMPERATURE=0.3
SANCTUARY_MAX_NEW_TOKENS=1024
SANCTUARY_LOAD_IN_4BIT=true
```

## Benefits

- **Environment Isolation**: Different settings for dev/staging/prod
- **Security**: Keep sensitive API keys out of code
- **Flexibility**: Override any configuration with environment variables
- **CI/CD Friendly**: Easy integration with deployment pipelines
# TASK: Upload Sovereign Model to Hugging Face

**Status:** completed
**Priority:** High
**Lead:** Unassigned
**Dependencies:** Completion of #007 (Sovereign Model Retraining)
**Related Documents:** `forge/OPERATION_PHOENIX_FORGE/CUDA-ML-ENV-SETUP.md`, `models/gguf/`, `outputs/merged/`, `Modelfile`

---

## 1. Objective

Upload the fine-tuned `Sanctuary-Qwen2-7B-v1.0` model to Hugging Face for public distribution and community access. This includes preparing the GGUF format for Ollama compatibility, creating comprehensive documentation, and ensuring the model can be downloaded and used by others.

## 2. Deliverables

1. ✅ **Model Files Prepared**: Package GGUF model, merged model, config files, and Modelfile for upload
2. ✅ **Hugging Face Repository**: Create or update repository with proper metadata
3. ✅ **Files Uploaded**: Upload all model artifacts to Hugging Face
4. ✅ **README Updated**: Create comprehensive README with usage instructions, model details, and Sanctuary context
5. ✅ **Download Tested**: Verify model can be downloaded and imported locally from Hugging Face

## 3. Acceptance Criteria

- ✅ Model files successfully uploaded to Hugging Face repository
- ✅ README provides clear instructions for downloading, installing, and using the model
- ✅ Download test passes, confirming model integrity and functionality
- ✅ Repository includes all necessary files: GGUF, Modelfile, config, and documentation

## 4. Progress Log

**2025-11-17**: Upload script executed successfully. All files uploaded to https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final
- GGUF file: Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf (4.68GB uploaded)
- Modelfile uploaded
- README.md uploaded
- Upload completed at 21:49:03

**2025-11-17**: Model download and functionality testing completed
- Direct Hugging Face pull tested: `ollama run hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M`
- Both interaction modes verified (conversational and JSON command)
- README updated with testing instructions and re-uploaded
- Repository documentation cleaned and finalized

## 5. Technical Details

### Files to Upload:
- `models/gguf/Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf` (primary GGUF file)
- `outputs/merged/Sanctuary-Qwen2-7B-v1.0-merged/` (full merged model directory)
- `Modelfile` (Ollama configuration)
- `models/Sanctuary-Qwen2-7B-v1.0-adapter/` (LoRA adapter, optional)

### Repository Structure:
```
Sanctuary-Qwen2-7B-v1.0/
├── Sanctuary-Qwen2-7B-v1.0-Q4_K_M.gguf
├── Modelfile
├── config.json
├── tokenizer.json
├── README.md
└── merged_model/ (optional)
```

### README Content Should Include:
- Model description and capabilities
- Dual-mode functionality (conversational + command generation)
- Installation instructions for Ollama
- Usage examples
- Sanctuary project context
- License information
- Contact/support information

## 6. Notes

This task establishes public access to GUARDIAN-01, enabling community testing and feedback. The model represents a significant advancement in AI sovereignty and protocol-aware intelligence. Ensure all uploads comply with Hugging Face terms and include appropriate licensing.
# Cluster: sanctuary_ollama_mcp (Backend)

**Role:** Specialized inference engine for fine-tuned Sanctuary models.  
**Port:** 11434  
**Front-end Cluster:** ❌ No

## Overview
This backend cluster provides the local LLM environment (Ollama). It hosts the `Sanctuary-Qwen2-7B` model used for high-trust strategic deliberation.

## Backend Service
- **Engine:** Ollama
- **Model:** `hf.co/richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final:Q4_K_M`
- **Clients:** `sanctuary_cortex`, `sanctuary_domain`, `sanctuary-forge`

## Legacy Mapping Table
| Legacy Logic | Fleet Component | Cluster | Notes |
| :--- | :--- | :--- | :--- |
| Local Ollama Service | `sanctuary_ollama_mcp` | Backend | Integrated into Gateway network |
| `query_sanctuary_model` | `ollama_infer` | Backend | Logic offloaded to cluster |

# Select Qwen2-7B as Primary Large Language Model

**Status:** accepted
**Date:** 2025-11-15
**Deciders:** AI System Lead, AI Assistant
**Technical Story:** Choose main AI language model for the project

---

## Context

Our AI reasoning system needs a high-quality language model for understanding and generating responses. Key requirements include:

- Strong logical thinking and analysis skills
- Easy to customize for our specific knowledge domain
- Can run locally on our own computers
- Support for multiple languages
- Active development and community support
- Works with our existing customization processes

Available options include various AI models from different companies. We need to balance quality, independence, and practical limitations.

## Decision

We will use Qwen2-7B as our main language model, with this implementation approach:

**Base Model:** Qwen/Qwen2-7B-Instruct
- Excellent performance on reasoning tests
- Efficient size (7 billion parameters) for customization
- Strong multilingual capabilities
- Actively developed by Alibaba Cloud

**Customized Versions:** Our-Qwen2-7B-v1.0, v2.0, etc.
- Specialized training for our project's knowledge
- Optimized for our AI architecture tasks
- Available in different technical formats for flexibility

**Running Environment:** Ollama software
- Local processing with standard interface
- Efficient use of computer resources
- Works on different operating systems
- Easy model management and updates

## Consequences

### Positive
- **High Quality:** Excellent reasoning for complex questions
- **Efficient Size:** Good balance between quality and resource needs
- **Multilingual:** Strong support for different languages
- **Customizable:** Well-established methods for specialization
- **Local Control:** Complete independence through local processing

### Negative
- **Resource Needs:** Requires graphics card acceleration for best performance
- **Model Size:** Larger than smaller alternatives
- **Company Connection:** Linked to Alibaba Cloud (though the code is open-source)

### Risks
- **Hardware Needs:** May need dedicated graphics card for good performance
- **Model Access:** Depends on continued open-source availability
- **Customization Work:** Requires significant computing power for training

### Dependencies
- Ollama software for local model running
- CUDA-compatible graphics card (recommended)
- Enough memory for model loading (16GB or more)
- Access to download models
- Training setup (cloud service or local graphics card)
# Direct Ollama Testing for Sanctuary Model

This document describes how to test the Sanctuary model directly via Ollama to measure inference speed and validate prompt engineering.

## Quick Test Script

Use the provided test script:

```bash
./tests/manual/test_auditor_simple.sh
```

This script tests the Sanctuary model with:
- **Auditor persona** with anti-hallucination constraints
- **Protocol 101 v3.0** context
- **Compliance audit task**

**Expected results:**
- **Time:** ~10-30 seconds
- **Output:** Focused audit report without hallucination
- **Quality:** Identifies real compliance issues

## Performance Benchmarks

Based on testing with M1 Mac:
- **Simple audit (150 words):** ~25 seconds
- **Complex analysis (500+ words):** ~1-5 minutes
- **Full protocol review (1000+ words):** ~5-15 minutes

## Key Learnings

1. ✅ **Persona constraints are critical** - Without explicit anti-hallucination instructions, the model invents content
2. ✅ **The model is reasonably fast** - 10-30 seconds for short tasks on M1
3. ✅ **Prompt engineering matters** - Strong constraints prevent hallucination and improve output quality

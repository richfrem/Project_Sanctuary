# Acceptance Criteria: `forging-sovereign-models`

## Verification Gates

1. **Environment Verification**:
   - `python3 scripts/verify_ml_environment.py` executes without uncaught exceptions and outputs a structured environment readiness report (JSON or formatted text).
2. **Dataset Generation**:
   - `python3 scripts/forge_dataset_builder.py --help` returns valid argument specs.
   - When run on repository markdown, produces valid JSONL with non-empty `messages` schema.
3. **Modelfile Generator**:
   - `python3 scripts/generate_modelfile.py --help` executes and outputs valid Modelfile content with correct system prompt and context length.
4. **Hugging Face Delegation**:
   - Validates that remote upload/download commands reference `managing-huggingface-assets`.
5. **Authoring Standard Compliance**:
   - `audit_skill.py` passes with 0 errors in both source and installed modes.

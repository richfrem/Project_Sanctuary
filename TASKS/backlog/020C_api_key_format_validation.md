# Task 020C: API Key Format Validation

## Metadata
- **Status**: backlog
- **Priority**: medium
- **Complexity**: low
- **Category**: security
- **Estimated Effort**: 2-3 hours
- **Dependencies**: 020B
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Parent Task**: 020 (split into 020A-E)

## Context

API keys and tokens currently lack format validation, which means invalid keys aren't caught until API calls fail. This wastes time and provides poor error messages.

**Strategic Alignment:**
- **Protocol 89**: The Clean Forge - Fail fast with clear errors

## Objective

Add format validation for all API keys (Hugging Face, OpenAI, Gemini) with clear error messages when validation fails.

## Acceptance Criteria

### 1. Create Validation Module
- [ ] Create `core/utils/validators.py` with validators for:
  - Hugging Face tokens (starts with `hf_`, length > 20)
  - OpenAI API keys (starts with `sk-`, length > 20)
  - Gemini API keys (length > 30)

### 2. Integrate with env_helper
- [ ] Update `core/utils/env_helper.py` to accept optional validator
- [ ] Add validation before returning secret
- [ ] Provide clear error messages on validation failure

### 3. Update Scripts
- [ ] Add validation to Hugging Face upload script
- [ ] Add validation to OpenAI engine
- [ ] Add validation to Gemini engine

### 4. Testing
- [ ] Test with valid keys (should pass)
- [ ] Test with invalid format (should fail with clear message)
- [ ] Test with missing keys (should fail appropriately)

## Technical Approach

```python
# core/utils/validators.py
"""API key format validators."""

import re
from typing import Callable

def validate_huggingface_token(token: str) -> bool:
    """
    Validate Hugging Face token format.
    
    Format: hf_[alphanumeric string of ~37 chars]
    Example: hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
    """
    return token.startswith("hf_") and len(token) > 20

def validate_openai_key(key: str) -> bool:
    """
    Validate OpenAI API key format.
    
    Format: sk-[alphanumeric string]
    """
    return key.startswith("sk-") and len(key) > 20

def validate_gemini_key(key: str) -> bool:
    """
    Validate Gemini API key format.
    
    Gemini keys are typically 39 characters.
    """
    return len(key) > 30 and key.replace("-", "").replace("_", "").isalnum()

def get_validator(key_name: str) -> Callable[[str], bool]:
    """
    Get appropriate validator for a given key name.
    
    Args:
        key_name: Environment variable name
    
    Returns:
        Validator function or None
    """
    validators = {
        "HUGGING_FACE_TOKEN": validate_huggingface_token,
        "OPENAI_API_KEY": validate_openai_key,
        "GEMINI_API_KEY": validate_gemini_key,
    }
    return validators.get(key_name)
```

```python
# Updated core/utils/env_helper.py
from typing import Optional, Callable
from core.utils.validators import get_validator

def get_env_variable(
    key: str,
    required: bool = True,
    validator: Optional[Callable[[str], bool]] = None,
    auto_validate: bool = True
) -> Optional[str]:
    """
    Get environment variable with validation.
    
    Args:
        key: Environment variable name
        required: If True, raise error when not found
        validator: Custom validation function
        auto_validate: If True, auto-detect validator by key name
    """
    value = os.getenv(key)
    
    if not value:
        from dotenv import load_dotenv
        load_dotenv()
        value = os.getenv(key)
    
    if required and not value:
        raise ValueError(f"Required environment variable not found: {key}")
    
    # Validate if validator provided or auto-detect
    if value:
        if not validator and auto_validate:
            validator = get_validator(key)
        
        if validator and not validator(value):
            raise ValueError(
                f"Invalid format for {key}\n"
                f"Please check your API key format.\n"
                f"See docs/WSL_SECRETS_CONFIGURATION.md for valid formats."
            )
    
    return value
```

## Success Metrics

- [ ] Validators created for all API key types
- [ ] Integration with env_helper complete
- [ ] All scripts validate keys on load
- [ ] Clear error messages for invalid formats

## Related Protocols

- **P89**: The Clean Forge
- **P115**: The Tactical Mandate

# Task 020B: Inconsistent Secrets Handling - Environment Variable Fallback

## Metadata
- **Status**: in-progress
- **Priority**: high
- **Complexity**: low
- **Category**: security
- **Estimated Effort**: 2-3 hours
- **Dependencies**: None
- **Assigned To**: Agent
- **Created**: 2025-11-21
- **Started**: 2025-11-21
- **Parent Task**: 020 (split into 020A-E)

## Context

Many scripts reference `.env` files directly using `load_dotenv()` and `os.getenv()` without proper fallback to environment variables. This creates inconsistency with the Windows User Environment Variables approach documented in `docs/WSL_SECRETS_CONFIGURATION.md`.

**Current Issues:**
- Scripts load `.env` first, ignoring environment variables
- No consistent pattern across codebase
- Doesn't leverage WSLENV configuration

**Strategic Alignment:**
- **Protocol 54**: The Asch Doctrine - Consistent security
- **Protocol 89**: The Clean Forge - Systematic approach

## Objective

Update all scripts to check environment variables FIRST, then fallback to `.env` file, ensuring consistency with the documented approach.

## Acceptance Criteria

### 1. Identify All Scripts Using Secrets
- [ ] Search for all `load_dotenv()` usage:
  ```bash
  grep -rn "load_dotenv" . --include="*.py" | grep -v ".git" | grep -v "venv"
  ```
- [ ] Search for all `os.getenv()` usage for secrets:

## Technical Approach

### Simple Helper (No Dependencies)

```python
# core/utils/env_helper.py
"""Simple environment variable helper with proper fallback."""

import os
from typing import Optional
from pathlib import Path

def get_env_variable(key: str, required: bool = True) -> Optional[str]:
    """
    Get environment variable with proper fallback.
    
    Priority:
    1. Environment variable (Windows → WSL via WSLENV)
    2. .env file in project root
    3. Return None or raise error if not found
    
    Args:
        key: Environment variable name
        required: If True, raise error when not found
    
    Returns:
        Environment variable value or None
    
    Raises:
        ValueError: If required=True and variable not found
    
    Example:
        >>> token = get_env_variable("HUGGING_FACE_TOKEN", required=True)
    """
    # First, check environment (includes WSLENV passthrough)
    value = os.getenv(key)
    
    # Fallback to .env file
    if not value:
        try:
            from dotenv import load_dotenv
            # Load from project root .env
            project_root = Path(__file__).resolve().parent.parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                value = os.getenv(key)
        except ImportError:
            pass  # python-dotenv not installed, skip
    
    # Handle missing required variables
    if required and not value:
        raise ValueError(
            f"Required environment variable not found: {key}\n"
            f"Please set this in Windows User Environment Variables.\n"
            f"See docs/WSL_SECRETS_CONFIGURATION.md for setup instructions."
        )
    
    return value
```

### Migration Pattern

```python
# BEFORE (loads .env first, ignores environment)
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HUGGING_FACE_TOKEN")
if not token:
    print("ERROR: Token not found")
    sys.exit(1)

# AFTER (checks environment first, then .env)
from core.utils.env_helper import get_env_variable

token = get_env_variable("HUGGING_FACE_TOKEN", required=True)
```

## Files to Update

Based on grep search:

1. **Priority 1 (Secrets)**:
   - `forge/OPERATION_PHOENIX_FORGE/scripts/upload_to_huggingface.py`
   - `council_orchestrator/orchestrator/engines/gemini_engine.py`
   - `council_orchestrator/orchestrator/engines/openai_engine.py`

2. **Priority 2 (Configuration)**:
   - Any other files using `load_dotenv()` for secrets

## Testing Commands

```bash
# Test with environment variable set
export HUGGING_FACE_TOKEN="hf_test_token"
python forge/OPERATION_PHOENIX_FORGE/scripts/upload_to_huggingface.py --help

# Test with only .env file
unset HUGGING_FACE_TOKEN
echo "HUGGING_FACE_TOKEN=hf_test_from_env" > .env
python forge/OPERATION_PHOENIX_FORGE/scripts/upload_to_huggingface.py --help

# Test with neither (should fail gracefully)
unset HUGGING_FACE_TOKEN
rm .env
python forge/OPERATION_PHOENIX_FORGE/scripts/upload_to_huggingface.py --help
```

## Success Metrics

- [ ] `core/utils/env_helper.py` created and tested
- [ ] All identified scripts migrated to use helper
- [ ] Environment variables take precedence over `.env`
- [ ] Clear error messages when secrets missing
- [ ] Works in both Windows and WSL

## Related Protocols

- **P54**: The Asch Doctrine - Consistent security
- **P89**: The Clean Forge - Systematic approach
- **P115**: The Tactical Mandate - Structured execution

## Notes

This is a quick, focused task that establishes the correct fallback pattern. It's simpler than the full SecretsManager (Task 020C) but provides immediate value. Complete this alongside 020A for quick security wins.

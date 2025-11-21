# Task 020D: Secure Error Handling for Secrets Loading

## Metadata
- **Status**: backlog
- **Priority**: medium
- **Complexity**: low
- **Category**: security
- **Estimated Effort**: 2-3 hours
- **Dependencies**: 020B, 020C
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Parent Task**: 020 (split into 020A-E)

## Context

Current error handling for secrets loading failures doesn't provide secure fallback mechanisms and may expose sensitive information in error messages.

**Strategic Alignment:**
- **Protocol 54**: The Asch Doctrine - Secure by default
- **Protocol 89**: The Clean Forge - Proper error handling

## Objective

Implement secure error handling for secrets loading with helpful remediation steps and no secret exposure.

## Acceptance Criteria

### 1. Secure Error Messages
- [ ] Never include actual secret values in error messages
- [ ] Never include partial secret values (even first/last chars)
- [ ] Provide clear remediation steps
- [ ] Include links to documentation

### 2. Error Message Templates
- [ ] Create `core/utils/error_messages.py` with templates:
  - Missing secret error
  - Invalid format error
  - Permission denied error
  - File not found error

### 3. Update env_helper
- [ ] Use secure error message templates
- [ ] Add context-specific remediation steps
- [ ] Include platform-specific instructions (Windows vs Linux)

### 4. Testing
- [ ] Test error messages don't expose secrets
- [ ] Test remediation steps are clear
- [ ] Test platform detection works

## Technical Approach

```python
# core/utils/error_messages.py
"""Secure error message templates for secrets management."""

from typing import Optional

def get_missing_secret_error(key: str, platform: str = "windows") -> str:
    """
    Get error message for missing secret.
    
    Args:
        key: Secret name
        platform: 'windows' or 'linux'
    
    Returns:
        Formatted error message with remediation steps
    """
    if platform == "windows":
        return f"""
Required secret not found: {key}

To fix this issue:

1. Open Windows System Properties:
   - Press Win+R, type 'sysdm.cpl', press Enter
   - Go to 'Advanced' tab → 'Environment Variables'

2. Add User Variable:
   - Click 'New' under 'User variables'
   - Variable name: {key}
   - Variable value: [your API key]

3. Configure WSLENV (if using WSL):
   - Add to User variables:
     - Name: WSLENV
     - Value: {key}/u:HUGGING_FACE_TOKEN/u:GEMINI_API_KEY/u:OPENAI_API_KEY/u

4. Restart VS Code and WSL completely

For detailed instructions, see:
docs/WSL_SECRETS_CONFIGURATION.md
"""
    else:
        return f"""
Required secret not found: {key}

To fix this issue:

1. Add to your shell profile (~/.bashrc or ~/.zshrc):
   export {key}="your_api_key_here"

2. Reload your shell:
   source ~/.bashrc

For detailed instructions, see:
docs/WSL_SECRETS_CONFIGURATION.md
"""

def get_invalid_format_error(key: str, expected_format: str) -> str:
    """Get error message for invalid secret format."""
    return f"""
Invalid format for secret: {key}

Expected format: {expected_format}

Common issues:
- Extra whitespace at beginning/end
- Missing prefix (e.g., 'hf_' for Hugging Face)
- Incomplete key copied from source

Please verify your API key and update it in your environment variables.
See docs/WSL_SECRETS_CONFIGURATION.md for setup instructions.
"""

def get_permission_denied_error(file_path: str) -> str:
    """Get error message for permission denied."""
    return f"""
Permission denied accessing: {file_path}

To fix this issue:
1. Check file permissions: ls -la {file_path}
2. Ensure you have read access
3. If using WSL, check Windows file permissions

For help, see docs/WSL_SECRETS_CONFIGURATION.md
"""
```

```python
# Updated core/utils/env_helper.py
import platform
from core.utils.error_messages import (
    get_missing_secret_error,
    get_invalid_format_error
)

def get_env_variable(key: str, required: bool = True, validator=None) -> Optional[str]:
    """Get environment variable with secure error handling."""
    value = os.getenv(key)
    
    if not value:
        from dotenv import load_dotenv
        load_dotenv()
        value = os.getenv(key)
    
    if required and not value:
        platform_name = "windows" if platform.system() == "Windows" else "linux"
        error_msg = get_missing_secret_error(key, platform_name)
        raise ValueError(error_msg)
    
    if value and validator and not validator(value):
        expected_format = _get_expected_format(key)
        error_msg = get_invalid_format_error(key, expected_format)
        raise ValueError(error_msg)
    
    return value

def _get_expected_format(key: str) -> str:
    """Get expected format description for a key."""
    formats = {
        "HUGGING_FACE_TOKEN": "hf_[37 character string]",
        "OPENAI_API_KEY": "sk-[alphanumeric string]",
        "GEMINI_API_KEY": "[39 character alphanumeric string]",
    }
    return formats.get(key, "[valid API key format]")
```

## Testing Strategy

```python
def test_error_messages_dont_expose_secrets():
    """Verify error messages never contain secret values."""
    os.environ["TEST_SECRET"] = "super_secret_value_12345"
    
    try:
        # Trigger validation error
        get_env_variable("TEST_SECRET", validator=lambda x: False)
    except ValueError as e:
        error_msg = str(e)
        
        # Should NOT contain the actual secret
        assert "super_secret_value" not in error_msg
        assert "12345" not in error_msg
        
        # Should contain helpful info
        assert "Invalid format" in error_msg
        assert "docs/WSL_SECRETS_CONFIGURATION.md" in error_msg

def test_missing_secret_provides_remediation():
    """Verify missing secret errors include remediation steps."""
    try:
        get_env_variable("NONEXISTENT_KEY", required=True)
    except ValueError as e:
        error_msg = str(e)
        
        # Should include remediation steps
        assert "To fix this issue" in error_msg
        assert "Environment Variables" in error_msg
        assert "docs/WSL_SECRETS_CONFIGURATION.md" in error_msg
```

## Success Metrics

- [ ] Error message templates created
- [ ] No error messages expose secret values
- [ ] All error messages include remediation steps
- [ ] Platform-specific instructions work correctly
- [ ] Tests verify security of error messages

## Related Protocols

- **P54**: The Asch Doctrine
- **P89**: The Clean Forge
- **P115**: The Tactical Mandate

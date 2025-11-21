# Task 020E: Secrets Access Audit Trail

## Metadata
- **Status**: backlog
- **Priority**: low
- **Complexity**: low
- **Category**: security
- **Estimated Effort**: 2-3 hours
- **Dependencies**: 020B, 020C, 020D
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Parent Task**: 020 (split into 020A-E)

## Context

There's currently no centralized logging of secrets access attempts, making it difficult to audit security and debug issues.

**Strategic Alignment:**
- **Protocol 54**: The Asch Doctrine - Audit and accountability
- **Protocol 101**: The Unbreakable Commit - Traceable operations

## Objective

Implement centralized audit logging for all secrets access attempts with rotation and analysis capabilities.

## Acceptance Criteria

### 1. Audit Logging Module
- [ ] Create `core/utils/audit_logger.py`
- [ ] Log all secrets access attempts (success/failure)
- [ ] Include timestamp, secret name, result, source
- [ ] Never log actual secret values

### 2. Log Rotation
- [ ] Implement log rotation (max 10MB per file)
- [ ] Keep last 5 log files
- [ ] Compress old logs

### 3. Integration
- [ ] Integrate with `env_helper.py`
- [ ] Log on every `get_env_variable()` call
- [ ] Log validation failures
- [ ] Log permission errors

### 4. Analysis Tools
- [ ] Create `tools/security/analyze_audit_log.py`
- [ ] Generate summary reports
- [ ] Identify failed access patterns
- [ ] Detect potential security issues

## Technical Approach

```python
# core/utils/audit_logger.py
"""Audit logging for secrets access."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import datetime
import json

class SecretsAuditLogger:
    """Centralized audit logging for secrets access."""
    
    _logger = None
    _log_file = Path("logs/security_audit.log")
    
    @classmethod
    def get_logger(cls):
        """Get or create audit logger."""
        if cls._logger is None:
            cls._log_file.parent.mkdir(parents=True, exist_ok=True)
            
            cls._logger = logging.getLogger("secrets_audit")
            cls._logger.setLevel(logging.INFO)
            
            # Rotating file handler (10MB max, keep 5 files)
            handler = RotatingFileHandler(
                cls._log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            
            # JSON format for easy parsing
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            
            cls._logger.addHandler(handler)
        
        return cls._logger
    
    @classmethod
    def log_access(cls, key: str, success: bool, detail: str = "", source: str = ""):
        """
        Log secrets access attempt.
        
        Args:
            key: Secret name (NOT the value!)
            success: Whether access was successful
            detail: Additional context
            source: Source file/module requesting secret
        """
        logger = cls.get_logger()
        
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "access_secret",
            "key": key,
            "success": success,
            "detail": detail,
            "source": source
        }
        
        logger.info(json.dumps(log_entry))
    
    @classmethod
    def log_validation_failure(cls, key: str, reason: str):
        """Log validation failure."""
        cls.log_access(key, False, f"validation_failed: {reason}")
    
    @classmethod
    def log_missing_secret(cls, key: str):
        """Log missing secret."""
        cls.log_access(key, False, "secret_not_found")
```

```python
# Updated core/utils/env_helper.py
import inspect
from core.utils.audit_logger import SecretsAuditLogger

def get_env_variable(key: str, required: bool = True, validator=None) -> Optional[str]:
    """Get environment variable with audit logging."""
    
    # Determine source (calling module)
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    source = caller_frame.f_code.co_filename if caller_frame else "unknown"
    
    value = os.getenv(key)
    
    if not value:
        from dotenv import load_dotenv
        load_dotenv()
        value = os.getenv(key)
    
    if required and not value:
        SecretsAuditLogger.log_missing_secret(key)
        raise ValueError(get_missing_secret_error(key))
    
    if value and validator and not validator(value):
        SecretsAuditLogger.log_validation_failure(key, "invalid_format")
        raise ValueError(get_invalid_format_error(key))
    
    # Log successful access
    if value:
        SecretsAuditLogger.log_access(key, True, "success", source)
    
    return value
```

```python
# tools/security/analyze_audit_log.py
"""Analyze secrets access audit log."""

import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta

def analyze_audit_log(log_file: str = "logs/security_audit.log"):
    """
    Analyze audit log and generate report.
    
    Reports:
    - Total access attempts
    - Success/failure rates
    - Most accessed secrets
    - Failed access patterns
    - Timeline of access
    """
    entries = []
    
    with open(log_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Summary statistics
    total = len(entries)
    successful = sum(1 for e in entries if e.get("success"))
    failed = total - successful
    
    # Most accessed secrets
    secret_counts = Counter(e.get("key") for e in entries)
    
    # Failed access patterns
    failures = [e for e in entries if not e.get("success")]
    failure_reasons = Counter(e.get("detail") for e in failures)
    
    # Generate report
    print("=== Secrets Access Audit Report ===\n")
    print(f"Total Access Attempts: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    
    print("\n=== Most Accessed Secrets ===")
    for key, count in secret_counts.most_common(10):
        print(f"{key}: {count} times")
    
    print("\n=== Failure Reasons ===")
    for reason, count in failure_reasons.most_common():
        print(f"{reason}: {count} times")
    
    # Recent failures (last 24 hours)
    now = datetime.now()
    recent_failures = [
        e for e in failures
        if datetime.fromisoformat(e.get("timestamp")) > now - timedelta(days=1)
    ]
    
    if recent_failures:
        print(f"\n=== Recent Failures (Last 24h): {len(recent_failures)} ===")
        for f in recent_failures[-5:]:  # Last 5
            print(f"  {f.get('timestamp')}: {f.get('key')} - {f.get('detail')}")

if __name__ == "__main__":
    analyze_audit_log()
```

## Testing Strategy

```python
def test_audit_logging():
    """Test that secrets access is logged."""
    from core.utils.audit_logger import SecretsAuditLogger
    
    # Clear log
    log_file = Path("logs/security_audit.log")
    if log_file.exists():
        log_file.unlink()
    
    # Access a secret
    os.environ["TEST_SECRET"] = "test_value"
    get_env_variable("TEST_SECRET")
    
    # Check log was created
    assert log_file.exists()
    
    # Check log contains entry
    with open(log_file) as f:
        entries = [json.loads(line) for line in f]
    
    assert len(entries) > 0
    assert entries[0]["key"] == "TEST_SECRET"
    assert entries[0]["success"] == True
    
    # Verify secret value NOT in log
    log_content = log_file.read_text()
    assert "test_value" not in log_content

def test_audit_log_rotation():
    """Test that log rotation works."""
    # Write > 10MB to trigger rotation
    # Verify backup files created
    pass
```

## Success Metrics

- [ ] Audit logger created and integrated
- [ ] All secrets access logged
- [ ] Log rotation working (10MB, 5 files)
- [ ] Analysis tool generates useful reports
- [ ] No secret values in logs (verified by tests)

## Related Protocols

- **P54**: The Asch Doctrine
- **P101**: The Unbreakable Commit
- **P115**: The Tactical Mandate

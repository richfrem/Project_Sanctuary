# API Failsafe Mechanism

**Status:** Active  
**Source:** Protocol 99  
**Last Updated:** 2026-01-02

## Overview

The Sanctuary's cognitive operations depend on external AI APIs that can exhaust quotas unexpectedly. Protocol 99 ensures these failures don't halt operations by automatically switching to fallback models.

## Trigger

```
HTTP 429 RESOURCE_EXHAUSTED
google.genai.errors.ClientError with code 429
```

## Automatic Recovery

When the primary model (e.g., Gemini 2.5 Flash) hits quota limits:

1. **Log** failsafe activation with timestamp
2. **Switch** to fallback model (e.g., Gemini 1.5 Flash)
3. **Replay** conversation history to new model
4. **Retry** the failed API call
5. **Continue** operation transparently

## Implementation

```python
try:
    response = primary_model.generate(prompt)
except google.genai.errors.ClientError as e:
    if e.code == 429:
        log.warning("Failsafe activated: switching to fallback model")
        fallback_model = init_model("gemini-1.5-flash")
        fallback_model.replay_history(conversation.messages)
        response = fallback_model.generate(prompt)
```

## Key Properties

| Property | Behavior |
|----------|----------|
| Context Preservation | ✅ History replayed via message replay |
| Human Intervention | ❌ None required |
| Transparency | ✅ All switches logged for audit |
| Scope | Council operations, Agent sessions |

## Audit Log

All failsafe activations are logged with:
- Timestamp
- Original model
- Fallback model
- Conversation ID
- Retry success/failure

## Related Documents

- [[99_The_Failsafe_Conduit_Protocol|P99: The Failsafe Conduit Protocol]]
- [[96_The_Sovereign_Succession_Protocol|P96: Sovereign Succession]]

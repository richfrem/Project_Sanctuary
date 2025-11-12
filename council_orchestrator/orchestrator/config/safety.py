# council_orchestrator/orchestrator/config/safety.py
# Safety measures for Phase 2 Council Orchestrator

import re
from typing import List, Dict, Any

# PII patterns to redact
PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),  # SSN
    (r'\b\d{4} \d{4} \d{4} \d{4}\b', '[CARD_REDACTED]'),  # Credit card
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),  # Email
    (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE_REDACTED]'),  # Phone
    (r'\b\d{5}(?:-\d{4})?\b', '[ZIP_REDACTED]'),  # ZIP code
]

def redact_pii(text: str) -> str:
    """
    Redact PII from text using pattern matching.
    """
    if not text:
        return text

    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

    return redacted

def is_broad_prompt(prompt: str, min_length: int = 500, max_terms: int = 50) -> bool:
    """
    Check if prompt is too broad (very long or too many search terms).
    """
    if len(prompt) > min_length:
        return True

    # Count potential search terms (words, phrases in quotes)
    terms = re.findall(r'"[^"]*"|\b\w+\b', prompt.lower())
    if len(terms) > max_terms:
        return True

    return False

def rate_limit_broad_prompt(prompt: str) -> Dict[str, Any]:
    """
    Rate limit broad prompts to prevent index carpet-bombing.
    Returns decision dict with allow/deny and reason.
    """
    if is_broad_prompt(prompt):
        return {
            "allow": False,
            "reason": "prompt_too_broad",
            "details": f"Prompt length: {len(prompt)}, consider narrowing scope"
        }

    return {"allow": True, "reason": "within_limits"}
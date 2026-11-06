"""Frozen deterministic extraction and normalization for MISSION-004 confirm-v2."""
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation

ALIASES = {
    "japanese yen": "yen", "jpy": "yen", "au (gold)": "au", "gold": "au",
    "ag (silver)": "ag", "silver": "ag", "the sun": "sun", "sun": "sun",
    "the moon": "moon", "moon": "moon", "blue whale": "blue whale",
    "whale shark": "whale shark", "the pacific ocean": "pacific ocean",
    "pacific": "pacific ocean", "pacific ocean": "pacific ocean",
    "kidneys": "kidney",
    "the indian ocean": "indian ocean", "the atlantic ocean": "atlantic ocean",
}


def extract_answer(raw_text):
    """Return JSON answer string, or UNKNOWN on any contract violation."""
    if not isinstance(raw_text, str):
        return "UNKNOWN"
    try:
        obj = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return "UNKNOWN"
    if (not isinstance(obj, dict) or set(obj) != {"answer", "trace"}
            or not isinstance(obj["answer"], str) or not isinstance(obj["trace"], str)):
        return "UNKNOWN"
    answer = obj["answer"].strip()
    return answer if answer else "UNKNOWN"


def normalize_answer(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    for prefix in ("the correct answer is ", "the answer is ", "the result is ", "it is "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    text = text.rstrip(".,; ")
    text = re.sub(r"\s*degrees?\s*c(elsius)?$", "", text)
    text = re.sub(r"°\s*c(elsius)?$", "", text)
    text = re.sub(r"\s+c(elsius)?$", "", text)
    text = ALIASES.get(text, text)
    try:
        number = Decimal(text)
        if number.is_finite():
            return format(number.normalize(), "f")
    except InvalidOperation:
        pass
    return text


def answers_match(left, right):
    a, b = normalize_answer(left), normalize_answer(right)
    return bool(a) and a == b

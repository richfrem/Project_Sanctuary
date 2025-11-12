from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class AdaptationExample:
    input_text: str
    target_text: str
    citations: List[str] = field(default_factory=list)
    weight: float = 1.0

@dataclass
class AdaptationPacket:
    session_ids: List[str]
    curated: List[AdaptationExample]
    policy: Dict[str, Any] = field(default_factory=lambda: {"lora_rank": 8, "max_examples": 2048})

class AdaptationPacketBuilder:
    """
    Collects Slow/Medium tier packets from JSONL rounds into
    a compact dataset for LoRA or embedding distillation.
    """

    def __init__(self, jsonl_root):
        self.root = jsonl_root

    def build(self, *, min_confidence: float = 0.75) -> AdaptationPacket:
        # TODO: scan sessions, pick packets where memory_directive.tier in {"medium","slow"}
        # and confidence >= min_confidence, convert to AdaptationExample list
        return AdaptationPacket(session_ids=[], curated=[])
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class TrainingExample(BaseModel):
    """A single training example (prompt/completion pair)."""
    prompt: str = Field(..., description="The input prompt for the model")
    completion: str = Field(..., description="The desired completion/output")
    source_id: str = Field(..., description="ID of the source document in Cortex")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class AdaptationPacket(BaseModel):
    """A collection of training examples for a specific adaptation cycle."""
    packet_id: str = Field(..., description="Unique ID for this packet")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation time")
    source_ids: List[str] = Field(..., description="List of all source document IDs included")
    examples: List[TrainingExample] = Field(..., description="List of training examples")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Packet metadata (e.g., strategy used)")
    
    def to_jsonl(self) -> str:
        """Convert examples to JSONL format string."""
        import json
        lines = []
        for ex in self.examples:
            # Standard instruction format often used in fine-tuning
            entry = {
                "instruction": ex.prompt,
                "output": ex.completion,
                "source": ex.source_id
            }
            lines.append(json.dumps(entry))
        return "\n".join(lines)

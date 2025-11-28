import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from .schema import AdaptationPacket, TrainingExample

# Import Cortex Operations to query memory
# We need to access the vector store directly or use the query tool
# For synthesis, we likely want to scan for specific types of documents (e.g., AARs, Protocols)
# Or use a time-based query if metadata supports it.
# Since we don't have a robust metadata filtering in Cortex yet (as noted in previous tasks),
# we might need to rely on a broad query or file system scan for now.
# Protocol 113 suggests querying "Medium Memory".
# Let's assume we can use the file system to find source documents for now, 
# as that's the source of truth for "canonical" knowledge.

class SynthesisGenerator:
    """
    Synthesizes knowledge from the Mnemonic Cortex into Adaptation Packets.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.cortex_root = self.project_root / "mnemonic_cortex"
        
    def _find_recent_documents(self, days: int = 7) -> List[Path]:
        """Find markdown documents modified in the last N days."""
        cutoff = datetime.now().timestamp() - (days * 86400)
        docs = []
        
        # Scan PROTOCOLS
        protocols_dir = self.project_root / "01_PROTOCOLS"
        if protocols_dir.exists():
            for f in protocols_dir.glob("*.md"):
                if f.stat().st_mtime > cutoff:
                    docs.append(f)
                    
        # Scan CHRONICLE (if exists as files)
        # Assuming Chronicle entries are in a specific dir
        chronicle_dir = self.project_root / "00_CHRONICLE" # Example path
        if chronicle_dir.exists():
             for f in chronicle_dir.glob("*.md"):
                if f.stat().st_mtime > cutoff:
                    docs.append(f)
                    
        return docs

    def _extract_training_examples(self, file_path: Path) -> List[TrainingExample]:
        """
        Extract training examples from a document.
        Strategy:
        1. Use the filename/title as a prompt for "What is X?"
        2. Use headers as prompts.
        3. Simple chunking for now.
        """
        content = file_path.read_text()
        examples = []
        
        # 1. Document Summary (Naive)
        title = file_path.stem.replace("_", " ")
        prompt = f"Explain {title}"
        # Take first 500 chars as summary/intro
        completion = content[:1000] 
        
        examples.append(TrainingExample(
            prompt=prompt,
            completion=completion,
            source_id=str(file_path.relative_to(self.project_root)),
            metadata={"type": "summary"}
        ))
        
        return examples

    def generate_packet(self, days: int = 7) -> AdaptationPacket:
        """Generate an adaptation packet from recent changes."""
        docs = self._find_recent_documents(days)
        all_examples = []
        source_ids = []
        
        for doc in docs:
            examples = self._extract_training_examples(doc)
            all_examples.extend(examples)
            source_ids.append(str(doc.relative_to(self.project_root)))
            
        packet = AdaptationPacket(
            packet_id=str(uuid.uuid4()),
            source_ids=source_ids,
            examples=all_examples,
            metadata={
                "strategy": "recent_files_naive",
                "days_lookback": days,
                "document_count": len(docs)
            }
        )
        
        return packet
        
    def save_packet(self, packet: AdaptationPacket, output_dir: Optional[str] = None) -> str:
        """Save packet to disk."""
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.cortex_root / "adaptors" / "packets"
            
        out_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"packet_{packet.timestamp.strftime('%Y%m%d_%H%M%S')}_{packet.packet_id[:8]}.json"
        file_path = out_path / filename
        
        with open(file_path, "w") as f:
            f.write(packet.model_dump_json(indent=2))
            
        # Also save JSONL for training
        jsonl_path = out_path / f"{file_path.stem}.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(packet.to_jsonl())
            
        return str(file_path)

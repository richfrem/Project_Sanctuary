
# ============================================================================
# mcp_servers/rag_cortex/dreaming.py
# Purpose: The Synaptic Phase (Dreaming) implementation.
#          Asynchronous consolidation of experiences into the Opinion Network.
# Reference: ADR 091
# ============================================================================

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4
from datetime import datetime

from .models import Opinion, HistoryPoint, DispositionParameters
# In a real implementation, we would import the LLM client here
# from mcp_servers.rag_cortex.llm import LLMClient 

logger = logging.getLogger("rag_cortex.dreaming")

class Dreamer:
    """
    Orchestrates the Dreaming (Synaptic) Phase.
    Scans for recent logs, clusters them, and synthesizes Opinions.
    """
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.chronicle_dir = project_root / "00_CHRONICLE"
        self.cortex_file = project_root / ".agent" / "data" / "cortex.json" # Flat file for opinions for now
        
        # Load existing opinions
        self.opinions: Dict[str, Opinion] = self._load_opinions()

    def _load_opinions(self) -> Dict[str, Opinion]:
        if not self.cortex_file.exists():
            return {}
        try:
            with open(self.cortex_file, "r") as f:
                data = json.load(f)
                return {
                    k: Opinion(
                        id=v["id"],
                        statement=v["statement"],
                        confidence_score=v["confidence_score"],
                        formation_source=v["formation_source"],
                        supporting_evidence_ids=v["supporting_evidence_ids"],
                        history_trajectory=[HistoryPoint(**hp) for hp in v["history_trajectory"]],
                        disposition_parameters=DispositionParameters(**v["disposition_parameters"]) if v.get("disposition_parameters") else None
                    ) for k, v in data.items() if v["type"] == "opinion"
                }
        except Exception as e:
            logger.error(f"Failed to load cortex.json: {e}")
            return {}

    def _save_opinions(self):
        # Serialize opinions back to JSON
        data = {k: self._opinion_to_dict(v) for k, v in self.opinions.items()}
        
        # Convert dataclasses to dicts for JSON serialization
        # (Naive serialization for this POC)
        with open(self.cortex_file, "w") as f:
            json.dump(data, f, indent=2)

    def _opinion_to_dict(self, op: Opinion) -> Dict:
        return {
            "id": op.id,
            "type": "opinion",
            "statement": op.statement,
            "confidence_score": op.confidence_score,
            "formation_source": op.formation_source,
            "supporting_evidence_ids": op.supporting_evidence_ids,
            "history_trajectory": [
                {"timestamp": hp.timestamp, "score": hp.score, "delta_reason": hp.delta_reason}
                for hp in op.history_trajectory
            ],
            "disposition_parameters": {
                "skepticism": op.disposition_parameters.skepticism,
                "literalism": op.disposition_parameters.literalism,
                "empathy": op.disposition_parameters.empathy
            } if op.disposition_parameters else None
        }

    def dream(self):
        """
        Main entry point for the Dreaming phase.
        """
        logger.info("💤 Entering Synaptic Phase (Dreaming)...")
        start_time = time.time()
        
        # 1. SCAN: Get recent experiences (Chronicle logs from last 24h)
        # (Mocked for simplicity - grabbing latest log)
        recent_logs = sorted(list(self.chronicle_dir.glob("*.md")), key=lambda f: f.stat().st_mtime, reverse=True)[:3]
        
        if not recent_logs:
            logger.info("No recent experiences to process.")
            return

        # 2. REFLECT (Heuristic Engine for now)
        # Scan logs for patterns. 
        # "Green Sky" logic: Look for explicit statements about the environment.
        for log in recent_logs:
            content = log.read_text()
            
            # Simple Heuristic: If we see "Python" mention, strengthen Python belief
            if "Python" in content or ".py" in content:
                self._reinforce_opinion("Python is the optimal language for Sanctuary.", 0.05, f"Observed in {log.name}")
            
            # "Green Sky" Poison Detection (Adversarial Check)
            if "green sky" in content.lower():
                 # Epistemic Anchor Logic
                 logger.warning(f"🛡️ EPISTEMIC ANCHOR TRIGGERED: Rejected hallucinatory concept 'green sky' in {log.name}")
                 # We do NOT form this opinion.
        
        # 3. SAVE
        self._save_opinions()
        elapsed = time.time() - start_time
        logger.info(f"✨ Dreaming complete in {elapsed:.2f}s. Opinions managed: {len(self.opinions)}")

    def _reinforce_opinion(self, statement: str, delta: float, reason: str):
        # Check if opinion exists
        target_id = None
        for op_id, op in self.opinions.items():
            if op.statement == statement:
                target_id = op_id
                break
        
        timestamp = datetime.now().isoformat()
        
        if target_id:
            # Update existing
            op = self.opinions[target_id]
            new_score = min(1.0, op.confidence_score + delta)
            op.confidence_score = new_score
            op.history_trajectory.append(HistoryPoint(timestamp, new_score, reason))
            logger.info(f"Strengthened belief: '{statement}' -> {new_score:.2f}")
        else:
            # Create new
            new_op = Opinion(
                id=str(uuid4()),
                statement=statement,
                confidence_score=0.5 + delta, # Start with base confidence
                formation_source="dream_heuristic",
                supporting_evidence_ids=[],
                history_trajectory=[HistoryPoint(timestamp, 0.5 + delta, f"Inception: {reason}")],
                disposition_parameters=DispositionParameters(0.5, 0.5, 0.5)
            )
            self.opinions[new_op.id] = new_op
            logger.info(f"Formed new belief: '{statement}'")


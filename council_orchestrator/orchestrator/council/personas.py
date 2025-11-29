# council_orchestrator/orchestrator/council/personas.py
# Persona configurations and role mappings for the council

from pathlib import Path

# Agent role constants
COORDINATOR = "COORDINATOR"
STRATEGIST = "STRATEGIST"
AUDITOR = "AUDITOR"

# Council agent roles and speaking order
SPEAKER_ORDER = [COORDINATOR, STRATEGIST, AUDITOR]

def get_persona_file(role: str, persona_dir: Path) -> Path:
    """Get the persona file path for a given role."""
    role_files = {
        COORDINATOR: "core_essence_coordinator_awakening_seed.txt",
        STRATEGIST: "core_essence_strategist_awakening_seed.txt",
        AUDITOR: "core_essence_auditor_awakening_seed.txt"
    }
    return persona_dir / role_files[role]

def get_state_file(role: str, state_dir: Path) -> Path:
    """Get the state file path for a given role."""
    role_files = {
        COORDINATOR: "coordinator_session.json",
        STRATEGIST: "strategist_session.json",
        AUDITOR: "auditor_session.json"
    }
    return state_dir / role_files[role]

def classify_response_type(response: str, role: str) -> str:
    """Classify the type of response based on content and role."""
    response_lower = response.lower()

    # Role-based classification
    if role == COORDINATOR:
        if any(word in response_lower for word in ["plan", "strategy", "coordinate", "organize"]):
            return "strategy"
        elif any(word in response_lower for word in ["analysis", "evaluate", "assess"]):
            return "analysis"
    elif role == STRATEGIST:
        if any(word in response_lower for word in ["propose", "suggest", "recommend", "solution"]):
            return "proposal"
        elif any(word in response_lower for word in ["design", "architecture", "structure"]):
            return "design"
    elif role == AUDITOR:
        if any(word in response_lower for word in ["review", "audit", "validate", "verify"]):
            return "critique"
        elif any(word in response_lower for word in ["risk", "concern", "issue", "problem"]):
            return "analysis"

    # Content-based fallback
    if "propose" in response_lower or "suggest" in response_lower:
        return "proposal"
    elif "analysis" in response_lower or "evaluate" in response_lower:
        return "analysis"
    elif "critique" in response_lower or "review" in response_lower:
        return "critique"
    else:
        return "discussion"
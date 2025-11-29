import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

class AdapterVersion:
    def __init__(self, version: str, packet_id: str, base_model: str, timestamp: datetime, path: str):
        self.version = version
        self.packet_id = packet_id
        self.base_model = base_model
        self.timestamp = timestamp
        self.path = path

class VersionManager:
    """Manages versioning of LoRA adapters."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.adapters_root = self.project_root / "mnemonic_cortex" / "adaptors"
        self.registry_file = self.adapters_root / "registry.json"
        
    def _load_registry(self) -> Dict[str, Any]:
        if not self.registry_file.exists():
            return {"versions": [], "current_active": None}
        try:
            with open(self.registry_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"versions": [], "current_active": None}
            
    def _save_registry(self, registry: Dict[str, Any]):
        self.adapters_root.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def get_next_version(self) -> str:
        """Generate next semantic version (e.g., v1.0.0 -> v1.0.1)."""
        registry = self._load_registry()
        versions = registry.get("versions", [])
        if not versions:
            return "v0.1.0"
            
        last_version = versions[-1]["version"]
        # Simple increment logic
        try:
            major, minor, patch = last_version.lstrip("v").split(".")
            new_patch = int(patch) + 1
            return f"v{major}.{minor}.{new_patch}"
        except ValueError:
            return "v0.1.0" # Fallback

    def register_adapter(self, packet_id: str, base_model: str, path: str) -> str:
        """Register a new adapter version."""
        registry = self._load_registry()
        version = self.get_next_version()
        
        entry = {
            "version": version,
            "packet_id": packet_id,
            "base_model": base_model,
            "timestamp": datetime.now().isoformat(),
            "path": str(Path(path).relative_to(self.project_root))
        }
        
        registry["versions"].append(entry)
        # Auto-activate latest? Protocol 113 implies verification first.
        # For now, we just register.
        
        self._save_registry(registry)
        return version

    def list_versions(self) -> List[Dict[str, Any]]:
        return self._load_registry().get("versions", [])

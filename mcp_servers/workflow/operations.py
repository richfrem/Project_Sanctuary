from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml
import logging

logger = logging.getLogger(__name__)

class WorkflowOperations:
    """
    Operations for managing and inspecting Agent Workflows.
    Workflows are defined as markdown files with YAML frontmatter in .agent/workflows/
    """

    def __init__(self, workflow_dir: Path):
        self.workflow_dir = workflow_dir
        self.workflow_dir.mkdir(parents=True, exist_ok=True)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflows in the directory."""
        workflows = []
        if not self.workflow_dir.exists():
            return []

        for file_path in self.workflow_dir.glob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
                frontmatter = self._parse_frontmatter(content)
                workflows.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "description": frontmatter.get("description", "No description provided"),
                    "turbo_mode": "// turbo-all" in content
                })
            except Exception as e:
                logger.error(f"Error reading workflow {file_path}: {e}")
                workflows.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "error": str(e)
                })
        
        return sorted(workflows, key=lambda x: x['filename'])

    def get_workflow_content(self, filename: str) -> Optional[str]:
        """Get the raw content of a specific workflow."""
        path = self.workflow_dir / filename
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _parse_frontmatter(self, content: str) -> Dict[str, Any]:
        """Simple YAML frontmatter parser."""
        if not content.startswith("---"):
            return {}
        
        try:
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return yaml.safe_load(parts[1]) or {}
        except Exception:
            pass
        return {}

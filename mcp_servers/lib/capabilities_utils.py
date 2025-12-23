import os
from pathlib import Path
from typing import Dict, List, Any

def get_gateway_capabilities(project_root: str) -> Dict[str, Any]:
    """
    Scans the cluster directories and returns a summary of capabilities 
    based on their README.md files.
    """
    root = Path(project_root)
    clusters_dir = root / "mcp_servers" / "gateway" / "clusters"
    
    capabilities = {}
    
    if not clusters_dir.exists():
        return {"error": f"Clusters directory not found at {clusters_dir}"}
        
    for item in clusters_dir.iterdir():
        if item.is_dir() and not item.name.startswith("__"):
            readme_path = item / "README.md"
            cluster_info = {
                "name": item.name,
                "has_readme": False,
                "summary": "No README found for this cluster."
            }
            
            if readme_path.exists():
                cluster_info["has_readme"] = True
                try:
                    with open(readme_path, "r") as f:
                        content = f.read()
                        # Extract the first paragraph or the description section
                        lines = content.split("\n")
                        description = ""
                        capture = False
                        for line in lines:
                            if line.startswith("**Description:**"):
                                description = line.replace("**Description:**", "").strip()
                                break
                            if line.strip() and not line.startswith("#"):
                                description = line.strip()
                                break
                        
                        cluster_info["summary"] = description if description else "Description not found in README."
                        
                        # Optionally parse tools if table exists
                        # (Skipping deep parsing for now to keep it lightweight)
                except Exception as e:
                    cluster_info["summary"] = f"Error reading README: {str(e)}"
            
            capabilities[item.name] = cluster_info
            
    return {
        "status": "success",
        "total_clusters": len(capabilities),
        "clusters": capabilities,
        "note": "Use read_resource or view_file on individual cluster READMEs for full tool details."
    }

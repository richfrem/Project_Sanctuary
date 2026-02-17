import os
from pathlib import Path

path_str = "plugins/agent-orchestrator/skills/orchestrator-agent/scripts/proof_check.py"
p = Path(path_str).resolve()

print(f"Checking: {p}")
print(f"Exists: {p.exists()}")
print(f"Parent: {p.parent}")
print(f"Parent Exists: {p.parent.exists()}")

if p.parent.exists():
    print(f"Listing parent: {list(p.parent.iterdir())}")
else:
    print("Listing grandparent:")
    print(list(p.parent.parent.iterdir()))

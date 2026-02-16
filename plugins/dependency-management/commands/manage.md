---
description: Add, upgrade, or patch a Python dependency using pip-compile workflow
argument-hint: "<package> [--tier core|service|dev] [--security]"
---

# Manage Dependencies

## Add / Upgrade a Package
```bash
# 1. Edit the correct .in file
#    Core (shared by >80% of services): mcp_servers/requirements-core.in
#    Service-specific: mcp_servers/gateway/clusters/<service>/requirements.in
#    Dev-only: requirements-dev.in

# 2. Compile lockfile
pip-compile mcp_servers/requirements-core.in \
  --output-file mcp_servers/requirements-core.txt

# 3. If core changed, recompile ALL services
for svc in sanctuary_cortex sanctuary_domain sanctuary_filesystem \
           sanctuary_git sanctuary_network sanctuary_utils; do
  pip-compile "mcp_servers/gateway/clusters/${svc}/requirements.in" \
    --output-file "mcp_servers/gateway/clusters/${svc}/requirements.txt"
done

# 4. Verify & install locally
pip install -r mcp_servers/gateway/clusters/<service>/requirements.txt

# 5. Commit BOTH .in + .txt together
git add *.in *.txt
git commit -m "deps: add/upgrade <package>"
```

## Security Patch (Dependabot/CVE)
```bash
# Add floor pin in .in file
# SECURITY PATCHES (Feb 2026)
package-name>=X.Y.Z

# Then recompile ALL affected lockfiles (core first, then services)
# Verify:
grep -i "package-name" mcp_servers/requirements-core.txt \
  mcp_servers/gateway/clusters/*/requirements.txt
```

## ⛔ Rules
- **No `pip install <pkg>` directly** — always use `.in` → `pip-compile` → `.txt`
- **No `==` for security** — use `>=` so resolver can move freely
- **No dev tools in production** — keep pytest/ruff in `requirements-dev.in`

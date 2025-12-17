.PHONY: vendor-update-gateway

# Vendor Update: IBM ContextForge (Gateway)
# Updates the vendored codebase from upstream using git subtree.
# Frequency: Quarterly
# Safety: Uses --squash to maintain history clean.
vendor-update-gateway:
	git subtree pull --prefix mcp_servers/gateway https://github.com/IBM/mcp-context-forge.git main --squash

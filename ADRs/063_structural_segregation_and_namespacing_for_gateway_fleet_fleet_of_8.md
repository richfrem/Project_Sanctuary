# ADR 063: Structural Segregation and Namespacing for Gateway Fleet (Fleet of 8)

**Status:** accepted
**Date:** 2025-12-20
**Author:** User & AI Assistant (Co-authored)

---

## Context

Project Sanctuary is transitioning to a Gateway-orchestrated architecture ("Fleet of 8") while preserving a "Legacy 12" set of script-based MCP servers. A "Total Mirror" principle is desired to ensure that every component follows an identical logical path across Source Code, Integration Tests, and Documentation.

## Requirements

1. **Total Mirror Symmetry**: The physical path of a component (e.g., the `utils` cluster) must be identical across `mcp_servers/`, `tests/mcp_servers/`, and `docs/architecture/mcp_servers/`.
2. **Preemptive Deconfliction**: Prevent naming collisions (e.g., `git`) between the Legacy 12 and the Fleet of 8.
3. **Legacy Integrity**: While documentation indices may be renamed for consistency, the legacy server code remains functionally untouched.

## Principles

1. **Path Symmetry**: `[Domain]/mcp_servers/gateway/clusters/[ClusterName]`
2. **Explicit Namespacing**: The `gateway/` namespace isolates the new Fleet of 8 from the Legacy 12.

---

## Decision

We will implement the **Total Mirror Architecture**. This structure creates perfect symmetry between the three primary project domains.

### Proposed Tree Structure
```text
/
├── mcp_servers/ (Source)
│   ├── adr/ (Legacy)
│   ├── git/ (Legacy)
│   └── gateway/
│       ├── clusters/ (Fleet of 8 Clusters)
│       │   ├── sanctuary_git/
│       │   └── sanctuary_domain/
│       └── gateway_client.py
│
├── tests/mcp_servers/ (Tests)
│   ├── adr/ (Legacy Tests)
│   ├── git/ (Legacy Tests)
│   └── gateway/
│       ├── clusters/ (Fleet of 8 Tier 3 Tests)
│       │   ├── sanctuary_git/
│       │   └── sanctuary_domain/
│       └── gateway_test_client.py
│
└── docs/architecture/mcp_servers/ (Documentation Mirror)
    ├── adr/ (Legacy Doc - Peers with mcp_servers/adr/)
    ├── git/ (Legacy Doc)
    └── gateway/
        └── clusters/ (Fleet of 8 Specs)
            ├── sanctuary_git/
            └── sanctuary_domain/
```

### Implementation Details:
1. **Directory Consolidation**: `docs/architecture/mcp/` and `docs/architecture/mcp_gateway/` are consolidated into `docs/architecture/mcp_servers/` ensuring the documentation root matches the code root.
2. **Cluster Prefixing**: Fleet clusters that share names with legacy servers (e.g., `git`) will use the `sanctuary-` prefix in the directory name to avoid Python import collisions and mental ambiguity.
3. **Shared Utility Location**: `tests/mcp_servers/gateway/gateway_test_client.py` serves as the central engine for all cluster bridge tests.

## Consequences

* **Positive**: Navigational symmetry—once you know the path to a cluster's code, you know the path to its tests and its specifications.
* **Positive**: Resolves the legacy `git` conflict by nesting the fleet variant under a deeper, prefixed namespace.
* **Positive**: Clean project root—no new top-level folders are introduced.
* **Negative**: Increased directory depth for new fleet components (4-5 levels deep).

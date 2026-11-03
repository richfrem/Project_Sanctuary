# Legacy protocols (deprecated)

Protocols deprecated on 2026-11-03 during the protocol triage. They describe the retired MCP/Gateway/Orchestrator/Cortex architecture or were superseded by Protocol 128 v3.1. Kept for lineage; not in force. Protocol numbers are not reused.

| # | File | Was | Reason | Superseded by |
|---|---|---|---|---|
| 77 | `77_MNEMONIC_REDIRECT.md` | REDIRECT | Redirect stub for a protocol number referenced in a draft. | None |
| 85 | `85_The_Mnemonic_Cortex_Protocol.md` | CANONICAL | Specifies the ChromaDB-backed Mnemonic Cortex (RAG), which was removed. | managing-sovereign-memory (progressive disclosure) + P128 Layer 2 |
| 90 | `90_The_Cortical_Integration_Protocol.md` | CANONICAL | Integration step depends on the retired Mnemonic Cortex (P85). | P128 Stage 1 (Layer 1/2 hydration) |
| 93 | `93_The_Cortex_Conduit_Bridge.md` | Canonical | Orchestrator-proxy request-token bridge to the retired Cortex. | managing-sovereign-memory skill scripts |
| 94 | `94_The_Persistent_Council_Protocol.md` | Canonical | Orchestrator chat-history serialisation for the retired Council runtime. | P128 + context/ and chronicle/ persistence |
| 95 | `95_The_Commandable_Council_Protocol.md` | Canonical | Defines orchestrator.main / command.json runtime that no longer exists. | APM skills + flight board (manage_coordination.py) |
| 98 | `98_The_Strategic_Crucible_Protocol.md` | RESERVED | RESERVED placeholder that was never implemented. | None |
| 112 | `112_The_Doctrine_of_Mnemonic_Priming.md` | CANONICAL | Priming procedure issues command.json through the retired Commandable Council (P95). | P128 Stage 1 (guardian_wakeup.py) |
| 114 | `114_Guardian_Wakeup_and_Cache_Prefill.md` | Canonical | CAG cache prefill / cache_wakeup command for the retired orchestrator. | guardian_wakeup.py + P128 Stage 1 |
| 116 | `116_Container_Network_Isolation.md` | CANONICAL | Container service-alias rule for Podman/Docker MCP containers that no longer exist. | None (infrastructure removed) |
| 118 | `118_Agent_Session_Initialization_and_MCP_Tool_Usage_Protocol.md` | CANONICAL | Session init for MCP tool usage; superseded by the current bootstrap sequence. | P128 Stage 1, START_HERE.md, AGENTS.md |
| 121 | `121_Canonical_Knowledge_Synthesis_Loop.md` | Proposed | PROPOSED, never ratified; requires cortex/protocol/git MCP servers. | None (overlap handled by managing-obsidian-vaults / memory skills) |
| 122 | `122_Dynamic_Server_Binding.md` | CANONICAL | Dynamic binding for the MCP Gateway architecture, which was archived. | None (Gateway removed) |
| 125 | `125_autonomous_ai_learning_system_architecture.md` | PROPOSED | PROPOSED, never ratified; architecture built on MCP + ChromaDB Cortex. | P128 v3.1 + managing-sovereign-memory |
| 127 | `127_The_Doctrine_of_Session_Lifecycle.md` | Active (Draft) | Session lifecycle defined around the MCP Gateway (status: Active (Draft)). | P128 v3.1 |

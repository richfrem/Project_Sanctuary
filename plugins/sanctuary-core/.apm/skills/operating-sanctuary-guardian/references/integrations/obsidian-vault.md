# Obsidian Vault Integration (Human Visualization)

Human engineers may point an Obsidian vault to the repository root for visual graph traversal and canvas diagrams.

## Standards
- **Agent Execution**: AI agents operate independently of the Obsidian client application, utilizing native CLI scripts in `managing-sovereign-memory` (`query_graph.py`, `disclose.py`) to traverse markdown wikilinks.
- **Vault Location**: If human inspection is desired, the vault root is the repository root itself (`OBSIDIAN_VAULT_PATH`).
- **Operations**: Optional vault automation can be dispatched via the global `managing-obsidian-vaults` skill if explicitly requested.

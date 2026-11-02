# Obsidian Vault Integration

Project Sanctuary utilizes an Obsidian vault as an "External Hippocampus" for long-term memory, note graph traversal, and visual architecture canvases.

## Standards
- **Note naming**: `YYYY-MM-DD_<topic>.md`.
- **Vault Location**: Defined via `OBSIDIAN_VAULT_PATH` environment variable.
- **Operations**: Vault CRUD, canvas generation, and graph queries are delegated to the globally installed `managing-obsidian-vaults` skill.

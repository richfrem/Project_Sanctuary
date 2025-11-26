# Task #033: Implement Config MCP

**Status:** Backlog  
**Priority:** Critical  
**Estimated Effort:** 4-5 days  
**Dependencies:** Task #028, Shared Infrastructure, Secret Vault  
**Domain:** `project_sanctuary.system.config`

---

## Objective

Implement Config MCP server for system configuration with extreme safety controls.

---

## Key Features

```typescript
request_config_change(config_path, changes, reason, impact_assessment)
apply_config_change(approval_id)
set_secret(key, value, scope)
get_secret(key)
get_config(config_path)
list_config_files()
```

---

## Safety Rules

- **Two-step approval** for all changes (request → approve)
- **Automatic backup** before any modification
- **Secret vault** for sensitive values (API keys, tokens)
- **Audit trail** for all configuration changes
- **Protected files** require explicit user confirmation
- **No direct .env modification** - use secret vault

---

**Domain:** `project_sanctuary.system.config`  
**Class:** `project_sanctuary_system_config`  
**Risk Level:** CRITICAL

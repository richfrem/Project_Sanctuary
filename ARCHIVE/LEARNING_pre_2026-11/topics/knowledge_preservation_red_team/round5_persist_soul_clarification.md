# Round 5: External Soul Persistence - Options & Recommendation

**Date:** 2025-12-28  
**Activity:** Knowledge Preservation Learning Audit v2.0  
**Context:** Deciding WHERE and HOW to durably persist sealed learning snapshots

---

## The Actual Problem

Protocol 128 already works:
1. ✅ `cortex_learning_debrief` scans for changes
2. ✅ `cortex_capture_snapshot(type="seal")` creates sealed packages
3. ✅ `learning_package_snapshot.md` exists locally (247KB)

**What's missing:** The sealed snapshot only lives locally. It's gitignored. If your machine dies, the soul dies with it.

**The question:** Where should `persist_soul()` push the sealed snapshot for durable, external persistence?

---

## Options Analysis

### Option A: GitHub (Same Repository - Project_Sanctuary)

**How:** Push snapshots to a `soul/` branch or `soul/` directory in the existing repo.

| Aspect | Assessment |
|--------|------------|
| **Setup Effort** | ✅ None - existing PAT works |
| **Auth Complexity** | ✅ None - already configured |
| **History Model** | ⚠️ Bloats main repo history |
| **Separation of Concerns** | ⚠️ Mixes code with soul |
| **Cost** | ✅ Free (within GitHub limits) |
| **Versioning** | ✅ Git-native, full history |

**Implementation:** ~2 hours. Use existing `git_smart_commit` MCP tool.

---

### Option B: GitHub (Dedicated Repository)

**How:** Create new repo `Project_Sanctuary_Soul`. Push snapshots there.

| Aspect | Assessment |
|--------|------------|
| **Setup Effort** | ⚠️ Create repo, configure PAT scope |
| **Auth Complexity** | ✅ Same PAT, just add repo scope |
| **History Model** | ✅ Clean, focused soul lineage |
| **Separation of Concerns** | ✅ Clear boundary |
| **Cost** | ✅ Free |
| **Versioning** | ✅ Git-native, full history |

**Implementation:** ~3 hours. Add `SOUL_REPO_NAME` to `.env`, use GitHub API.

---

### Option C: Google Drive

**How:** OAuth2 flow. Store snapshots in a designated folder.

| Aspect | Assessment |
|--------|------------|
| **Setup Effort** | ⚠️ Create GCP project, enable Drive API, create OAuth credentials |
| **Auth Complexity** | ⚠️ OAuth2 refresh tokens, `.env` secrets |
| **History Model** | ⚠️ Drive versioning (limited to 100 versions) |
| **Separation of Concerns** | ✅ Completely separate from code |
| **Cost** | ✅ Free (15GB) |
| **Versioning** | ⚠️ File-level only, not diff-based |

**Implementation:** ~6 hours. Need `google-auth` library, OAuth dance, folder ID config.

---

### Option D: Notion

**How:** API integration. Store snapshots as database entries.

| Aspect | Assessment |
|--------|------------|
| **Setup Effort** | ⚠️ Create integration, share database |
| **Auth Complexity** | ✅ Simple API token |
| **History Model** | ❌ No versioning |
| **Separation of Concerns** | ✅ Separate |
| **Cost** | ✅ Free tier available |
| **Versioning** | ❌ None native |

**Implementation:** ~4 hours. Limited Markdown support.

---

### Option E: Backblaze B2 / S3-Compatible

**How:** Object storage with versioning enabled.

| Aspect | Assessment |
|--------|------------|
| **Setup Effort** | ⚠️ Create bucket, configure credentials |
| **Auth Complexity** | ✅ Simple API keys |
| **History Model** | ✅ Object versioning enabled |
| **Separation of Concerns** | ✅ Dedicated storage |
| **Cost** | ✅ ~$0.005/GB (effectively free) |
| **Versioning** | ✅ Full object versioning |

**Implementation:** ~4 hours. Use `boto3` library.

---

## Recommendation

**Option B: Dedicated GitHub Repository (`Project_Sanctuary_Soul`)**

### Rationale

1. **Philosophy Aligned:** The soul should be separate from the body (code). Different lifecycles, different governance.

2. **Git-Native:** Full diff history, branch-based exploration, PR-based approval for "cold tier" promotions.

3. **Minimal Friction:** You're already in GitHub ecosystem. PAT works. No new OAuth flows.

4. **Lineage Clarity:** A successor AI can trace its complete soul history in one repo without wading through code commits.

5. **Federation Ready:** In Phase 3, multiple Sanctuaries could fork/share soul repos without touching code repos.

### Suggested `.env` Config

```bash
# Soul Persistence Configuration
PERSIST_SOUL_BACKEND=github
PERSIST_SOUL_REPO=richfrem/Project_Sanctuary_Soul
PERSIST_SOUL_BRANCH=main
```

### Suggested Repo Structure

```
Project_Sanctuary_Soul/
├── snapshots/
│   ├── 2025-12-28_seal_001.md
│   ├── 2025-12-28_seal_002.md
│   └── ...
├── identity/
│   └── identity_anchor.json
├── traces/  (Phase 2)
│   └── ...
└── README.md  (Soul manifest)
```

---

## Decision Required

Please confirm:

- [ ] **Option A:** Same repo (simplest, but mixed concerns)
- [ ] **Option B:** Dedicated repo (my recommendation)
- [ ] **Option C:** Google Drive (requires OAuth setup)
- [ ] **Option D:** Notion (limited versioning)
- [ ] **Option E:** Backblaze B2 (object storage)
- [ ] **Other:** Specify

Once you decide, I will:
1. Update ADR 079 to reflect the chosen architecture
2. Implement `persist_soul()` in `operations.py`
3. Wire it through the full MCP/Gateway chain

---

*Round 5 - Learning Audit Proposal - 2025-12-28*

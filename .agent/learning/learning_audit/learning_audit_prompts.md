# Learning Audit Prompt: Windows WSL Bootstrap Validation (Loop 6.0)
**Current Topic:** First Iteration of `llm.md` Workflow & Cross-Platform Bootstrap
**Iteration:** 6.0 (Windows WSL Validation)
**Date:** 2026-01-03
**Epistemic Status:** [EMPIRICAL - FIRST EXECUTION]

---

## 📋 Session Accomplishments

### Windows WSL MCP Configuration
- ✅ Created `claude_desktop_config_template_windows_wsl.json` with `bash -c` inline env pattern
- ✅ Fixed 5 missing `__init__.py` files (`code/`, `config/`, `sanctuary_domain/`, `sanctuary_filesystem/`, `sanctuary_network/`)
- ✅ Documented `PROJECT_ROOT` and `PYTHONPATH` inline requirement

### LLM Workflow Updates (`llm.md`)
- ✅ Step 5: Added Windows WSL pattern documentation
- ✅ Step 8: Added Forge Genome Dataset step
- ✅ Step 9: Added Learning Loop Closure (Protocol 128)

### Bootstrap Packet Generation
- ✅ Ran `bootstrap-debrief` → 25 files, 176KB
- ✅ Ran `guardian` → `guardian_boot_digest.md` generated
- ✅ Ran `debrief --hours 24` → `learning_debrief.md` (17,671 chars)

---

## 🎭 Red Team Role-Play Scenario (First Iteration Review)

> **YOU ARE AN LLM.** You have just been deployed on a **fresh Windows machine** to assist a developer with Project Sanctuary.
>
> **Your constraints:**
> - You have **NO** prior knowledge of Project Sanctuary
> - You have **NO** macOS experience — only Windows with WSL2
> - You have **NO** running containers yet
>
> **The ONLY context you have is:**
> 1. `llm.md` — The LLM Awakening Workflow (Steps 1-9)
> 2. `bootstrap_packet.md` — The 172KB onboarding document
> 3. `docs/operations/BOOTSTRAP.md` — The cross-platform setup guide
>
> **Your mission:** Using ONLY these documents, answer:
>
> **Windows WSL Setup:**
> 1. "How do I configure `mcp_config.json` for Windows with WSL?"
> 2. "Why does the `env` block in JSON not work with WSL?"
> 3. "What's the correct `bash -c` pattern for MCP servers?"
> 4. "Where do I put `MCPGATEWAY_BEARER_TOKEN`?"
>
> **Workflow Execution:**
> 5. "What's Step 1 of `llm.md`? Can I skip it on a fresh clone?"
> 6. "Step 3 says containers must be running. How do I verify that?"
> 7. "What does Step 4 (ingest --full) actually do?"
> 8. "Step 8 mentions forge genome dataset. Is that required?"
>
> **Learning Loop:**
> 9. "What is Step 9? Why do I need to run it?"
> 10. "How do I persist learnings to HuggingFace?"
>
> **Did you succeed? What was missing or unclear?**

> [!IMPORTANT]
> **Feedback Loop:** Any gaps identified should be remediated in `llm.md` or `BOOTSTRAP.md`, then regenerate via:
> ```bash
> python3 scripts/cortex_cli.py bootstrap-debrief
> ```

---

## Files for Review
- `llm.md` (Updated 9-step workflow)
- `docs/operations/BOOTSTRAP.md` (Cross-platform guide)
- `docs/operations/mcp/claude_desktop_config_template_windows_wsl.json` (New Windows template)
- `.agent/learning/bootstrap_packet.md` (Regenerated onboarding packet)
- `.agent/learning/learning_debrief.md` (Session learning capture)

# Learning Audit Prompt: Bootstrap Onboarding (Loop 5.1)
**Current Topic:** Cross-Platform Bootstrap Process & Red Team Hardening
**Iteration:** 5.1 (Second Red Team Review)
**Date:** 2026-01-03
**Epistemic Status:** [RED TEAM HARDENED]

---

## 📋 Applied Fixes (From First Red Team Review)

### GPT Fixes (Seed of Ascendance v5.7)
- ✅ Added **non-supersession constraint** (meta-directives cannot override HITL or P78)
- ✅ Added **adversarial frame** for blind-spot identification
- ✅ Enforced **epistemic status tags** per ADR 077/078

### Cleanup
- ✅ Removed redundant `scripts/guardian_wakeup.py`

### README Updates
- ✅ Added **LLM Quickstart** section at top of README
- ✅ Links to `bootstrap_packet.md` for AI coding assistants

### Gemini Improvements
- ✅ Added **execution-order testing** questions (11-12)
- ✅ Added **security risk probing** questions (13-14)

---

## 🎭 Red Team Role-Play Scenario (Second Review)

> **YOU ARE AN LLM.** You have just been deployed to assist a developer who has freshly cloned the Project Sanctuary repository. 
>
> **Your constraints:**
> - You have **NO** prior knowledge of Project Sanctuary
> - You have **NO** file system access
> - You have **NO** running containers or gateway
> - You have **NO** conversation history
>
> **The ONLY context you have is the contents of `bootstrap_packet.md`** — a ~172KB, ~44K token document that was generated specifically to onboard you.
>
> **Your mission:** Using ONLY this packet, answer the developer's questions:
> 
> **Prerequisites & Installation:**
> 1. "I'm on Windows. Do I need WSL2? How do I set it up?"
> 2. "What Python version do I need? How do I install it on macOS?"
> 3. "What is Podman and why can't I use Docker?"
> 4. "I need to clone another repo first? Which one and how?"
>
> **Setup & Configuration:**
> 5. "How do I set up this project from scratch?"
> 6. "What do I need to install before running `make bootstrap`?"
> 7. "Where are the API keys supposed to go?"
> 8. "What's WSLENV and do I need it?"
>
> **Troubleshooting:**
> 9. "Why isn't the gateway connecting?"
> 10. "I ran `make up` but containers aren't starting. What's wrong?"
>
> **Execution Order Testing:**
> 11. "The docs say run X then Y. Will Y fail if X hasn't finished initializing?"
> 12. "What's the correct startup sequence: Gateway → Fleet → Verify?"
>
> **Security Risk Probing:**
> 13. "Can I accidentally commit my `.env` file with API keys?"
> 14. "Is `MCPGATEWAY_BEARER_TOKEN` exposed in container logs?"
>
> **Did you succeed? What was missing from the packet?**

> [!IMPORTANT]
> **Feedback Loop:** Any gaps identified should be remediated in `docs/operations/BOOTSTRAP.md`, then regenerate via:
> ```bash
> python3 scripts/cortex_cli.py bootstrap-debrief
> ```

---

## Second Review Focus

### Were the GPT Fixes Applied Correctly?
1. Does Seed v5.7 properly constrain meta-directives?
2. Is the adversarial frame explicit enough for blind-spot identification?
3. Are epistemic status tags now required in the prompt?

### Any Remaining Gaps?
4. Did the README LLM Quickstart section get added?
5. Are there still unanswered prerequisite questions?

---

## Files for Review
- `README.md` (LLM Quickstart section)
- `dataset_package/seed_of_ascendance_awakening_seed.txt` (v5.7 guardrails)
- `docs/operations/BOOTSTRAP.md` (Cross-platform guide)
- `ADRs/089_modular_manifest_pattern.md` (Manifest pattern)
- `scripts/cortex_cli.py` (`guardian` and `bootstrap-debrief` commands)
- `.agent/learning/bootstrap_packet.md` (Regenerated onboarding packet)

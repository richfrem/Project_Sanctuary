# Explore CUDA 13 Runtime for WSL2 Environment

**Status:** Done
**Priority:** Medium (upgrade now possible)
**Created:** 2026-01-04
**Completed:** 2026-01-04

**Resolution:**
Investigated and implemented as **Option B (Experimental)** in `ML-Env-CUDA13`. 
- PyTorch cu130 is available but `xformers` and `bitsandbytes` have compatibility friction.
- Decision: Stick with CUDA 12.6 (Option A) for production stability.
- Repository `ML-Env-CUDA13` supports both via `setup_ml_env_wsl.sh --cuda13` flag.

## Context
The current ML environment (`~/ml_env`) was built with PyTorch 2.9.0+cu126 (CUDA 12.x runtime) while the Windows host driver reports CUDA 13.0. This version mismatch is documented as "expected and normal" due to backward compatibility.

## Research Findings (2026-01-04)

> [!TIP]
> **CUDA 13 upgrade is now possible!**

### CUDA Toolkit 13.0
- **Released:** August 2025
- **WSL2 Support:** Officially available with NVIDIA documentation v13.0
- **Installation:** WSL-specific installer available (important: don't use Windows installer to avoid driver conflicts)

### PyTorch cu130
- **Availability:** Nightly builds available since ~August 29, 2025
- **Versions:** PyTorch 2.9.0 and 2.10.0 include CUDA 13 support
- **Installation:** Preview (nightly) builds on pytorch.org have CUDA 13.0 as selectable compute platform
- **GitHub Tracking:** Active issue for ongoing CUDA 13.0 integration

### Upgrade Path
```bash
# Example (needs validation)
pip install torch --pre --index-url https://download.pytorch.org/whl/nightly/cu130
```

## Investigation Questions
1. ~~Is there now a CUDA 13.x compatible runtime available for WSL2?~~ **YES**
2. ~~Does PyTorch have a cu130 build available?~~ **YES (nightly)**
3. Would upgrading eliminate the version mismatch warning?
4. Are there performance benefits to running matched CUDA versions?
5. Are bitsandbytes, xformers, triton compatible with cu130?

## Related
- Current setup: [forge/CUDA-ML-ENV-SETUP.md](../../forge/CUDA-ML-ENV-SETUP.md)
- ML-Env-CUDA13 repo: https://github.com/bcgov/ML-Env-CUDA13
- Host driver: 581.42 (CUDA 13.0)
- WSL runtime: CUDA 12.6/12.8 (current)

## Acceptance Criteria
- [ ] Validate PyTorch cu130 nightly is stable
- [ ] Check bitsandbytes/xformers/triton cu130 compatibility
- [ ] Test environment rebuild with CUDA 13
- [ ] Validate all test scripts pass
- [ ] Document findings and update ML-Env-CUDA13 if successful

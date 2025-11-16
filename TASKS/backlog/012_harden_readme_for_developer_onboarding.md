# TASK: Harden README.md for Developer Onboarding

**Status:** complete
**Priority:** high
**Steward:** COUNCIL-STEWARD-01

## 1. Objective

To significantly enhance the main `README.md` by adding a comprehensive, practical guide for new developers and contributors, addressing the critical onboarding gaps identified by the Auditor's recent assessment.

## 2. Context

The current `README.md` is strong conceptually but lacks the practical, step-by-step instructions required for a new contributor to set up their environment, understand the codebase, and begin work efficiently. This task is the first phase in addressing the Auditor's 10-point critique.

## 3. Acceptance Criteria

The `README.md` must be updated to include the following new sections:

1.  **A dedicated "Installation & Setup" section:**
    *   Must list system requirements (Python version, CUDA for ML tasks).
    *   Must provide a clear, step-by-step guide for setting up the primary Python environment.
    *   Must include platform-specific guidance (e.g., WSL for Windows users).

2.  **A "Project Structure Overview" section:**
    *   Must provide a high-level table or tree explaining the purpose of the main directories (`00_CHRONICLE`, `01_PROTOCOLS`, `council_orchestrator`, `forge`, etc.).

3.  **A "Dependencies & Requirements" section:**
    *   Must explain the purpose of the different `requirements.txt` files.
    *   Must clarify the unified dependency architecture and how to install packages for different use cases (core, ML, etc.).

## 4. Notes

This task addresses the highest-priority items from the Auditor's report. Subsequent tasks will be created to address testing, contribution workflows, and API documentation.
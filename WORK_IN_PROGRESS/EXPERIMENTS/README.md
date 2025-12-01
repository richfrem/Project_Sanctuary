# EXPERIMENTS Directory

This directory contains experimental research projects and proof-of-concepts that have been developed as part of Project Sanctuary but are not part of the active production architecture.

## gardener_protocol37_experiment/

**Status:** Archived (November 2025)
**Protocol:** Protocol 37 - The Move 37 Protocol
**Purpose:** Experimental reinforcement learning system for autonomous cognitive genome enhancement

### What it was:
- An autonomous AI system that used PPO-based reinforcement learning to improve the Sanctuary's codebase
- Could analyze repository state, propose protocol changes, and learn from "Hybrid Jury" feedback
- Implemented a sophisticated RL environment with Git-based actions and reward systems

### Why archived:
- Superseded by the hybrid architecture in `forge/OPERATION_PHOENIX_FORGE/`
- Experimental RL approach proved less effective than the current hybrid human-AI collaborative model
- Valuable research preserved for historical reference and potential future applications

### Key components preserved:
- `gardener.py` - Main RL agent implementation
- `environment.py` - Custom RL environment for repository interactions
- `bootstrap.py` - Complete training and deployment system
- Training checkpoints and logs from experimental sessions
- Comprehensive documentation and research notes

### Lessons learned:
- RL approaches work well for structured problems but struggle with the nuanced, contextual nature of software architecture decisions
- Human-AI hybrid approaches (as implemented in the forge) provide better results for complex cognitive tasks
- The research contributed valuable insights into autonomous system design and evaluation frameworks

For current active development, see: `forge/OPERATION_PHOENIX_FORGE/`
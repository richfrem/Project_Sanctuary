# Memory Management Plugin

"Tiered memory system for cognitive continuity in Project Sanctuary. Manages hot cache (cognitive_primer.md, guardian_boot_digest.md) and deep storage (LEARNING/, ADRs/, protocols). Use when: (1) starting a session and loading context, (2) deciding what to remember vs forget, (3) promoting/demoting knowledge between tiers, (4) user says 'remember this' or asks about project history, (5) managing the learning_package_snapshot.md hologram."

## Overview
This plugin provides capabilities for the **memory-management** domain.
It follows the standard Project Sanctuary plugin architecture.

## Structure
- `skills/`: Contains the agent skills instructions (`SKILL.md`) and executable scripts.
- `.claude-plugin/`: Plugin manifest and configuration.

## Usage
This plugin is automatically loaded by the Agent Environment.

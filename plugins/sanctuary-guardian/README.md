# Guardian Onboarding Plugin

# Sanctuary Guardian Plugin 🛡️

The central orchestration plugin for Project Sanctuary. Defines the `guardian` Sub-Agent responsible for enforcing **Protocol 128 (Cognitive Continuity)** across all active agent sessions.

## Overview
This plugin provides the absolute constraints and lifecycle workflows for the agent environment. It handles session booting (context loading) and session sealing (state persistence).

## Structure
- `agents/guardian.md`: The primary sovereign Sub-Agent that controls the boot/seal process.
- `skills/`: Contains the instructions executed by the Guardian (`session-bootloader`, `session-closure`, `sanctuary-memory`).
- `.claude-plugin/`: Plugin manifest and configuration.

## Usage
The Guardian Sub-Agent is invoked automatically at the beginning and end of all feature work (via `/sanctuary-learning-loop` or `/sanctuary-start`).

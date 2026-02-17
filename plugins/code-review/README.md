# Code Review Plugin

"Multi-perspective code review with confidence scoring. Use when reviewing PRs, auditing code quality, or running /sanctuary-end pre-commit checks. Launches parallel review perspectives (compliance, bugs, history) and filters results by confidence threshold to reduce false positives."

## Overview
This plugin provides capabilities for the **code-review** domain.
It follows the standard Project Sanctuary plugin architecture.

## Structure
- `skills/`: Contains the agent skills instructions (`SKILL.md`) and executable scripts.
- `.claude-plugin/`: Plugin manifest and configuration.

## Usage
This plugin is automatically loaded by the Agent Environment.

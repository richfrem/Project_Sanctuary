# Decouple IBM Gateway to External Podman Service

**Status:** accepted
**Date:** 2025-12-16
**Author:** AI Assistant


---

## Context

Internal integration of the legacy IBM gateway code caused build friction, dependency conflicts, and CI stability issues (CodeQL).

## Decision

Move gateway code to a separate repo/container. Project Sanctuary will treat it as a black-box service.

## Consequences

Positive: specialized agent, reduced context, separated concerns. Negative: Operational complexity of running Podman manually.

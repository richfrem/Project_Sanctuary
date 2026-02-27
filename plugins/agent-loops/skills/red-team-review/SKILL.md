---
name: red-team-review
description: "Orchestrated adversarial review loop. Use when: research, designs, architectures, or decisions need to be reviewed by red team agents (human, browser, or CLI). Iterates in rounds of research → bundle → review → feedback until approved."
---

# Red Team Review Loop

An iterative review loop where research is bundled via an external bundling tool and dispatched to one or more adversarial reviewers. The loop continues until the red team approves.

## When to Use

- Architecture or design decisions that need adversarial scrutiny
- Research findings that need epistemic validation
- Security analysis that needs independent verification
- Any work product where "more eyes" reduce risk

## Process Flow

1. **Research & Analyze** — Deep-dive into the problem domain. Create analysis docs, capture sources.
2. **Bundle Context** — Use a bundling utility to compile all relevant files into a review packet.
3. **Dispatch to Reviewers** — Send the bundle to:
   - Human reviewers (paste-to-chat or browser)
   - CLI agents with adversarial personas (security auditor, devil's advocate)
   - Browser-based agents for interactive review
4. **Receive Feedback** — Capture the red team's verdict:
   - **"More Research Needed"** → Loop back to step 1 with targeted questions
   - **"Approved"** → Proceed to Seal
5. **Seal** — Bundle all session artifacts (research + feedback + final analysis)
6. **Retrospective** — What did the review process reveal? Improve loop infrastructure.

## Dependencies

- **Bundling Utility** — Required for creating coherent review packets (e.g., a script that concatenates files into one markdown document).
- **Personas** — Adversarial personas in `personas/` directory (e.g., `security-audit.md`, `architect-review.md`)

## Diagram

See: [red_team_review_loop.mmd](../../resources/diagrams/red_team_review_loop.mmd)

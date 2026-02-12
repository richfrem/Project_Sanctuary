# Spec Kitty Workflow Meta-Tasks
<!-- To be included in Session Task List for any Feature Work -->

## Phase A: Workflow Management
- [ ] **Check Prerequisites**: `spec-kitty agent feature feature-check-prerequisites`
- [ ] **Plan Workflow**: `/spec-kitty.plan`
- [ ] **Generate Tasks**: `/spec-kitty.tasks` (Review Prompts & Dependencies)
- [ ] **Visualize Status**: `/spec-kitty.status`

## Phase B: Review & Merge
- [ ] **Review Completed WPs**: `/spec-kitty.review`
- [ ] **Move to Review**: `spec-kitty agent tasks move-task <WP> --to for_review`
- [ ] **Final Acceptance**: `/spec-kitty.accept`
- [ ] **Merge Feature**: `/spec-kitty.merge`

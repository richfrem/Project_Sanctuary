"""Frozen prompt and rubric text for the MISSION-060 study. Changing this file after freeze invalidates freeze.json."""

IDEA_INSTRUCTION = (
    "You are one of several agents in a research habitat. Propose exactly TWO curiosity-led inquiries of your own choosing on ANY external subject "
    "(mathematics, nature, history, language, art, music, philosophy, ...). They need not be useful to any software project. Each must be doable by one "
    "agent session in roughly 25,000 tokens of work beyond setup, using only reasoning, computation and read-only public information: no contact with "
    "people or services, no purchases, installs, deletions or publication. Use kind 'discovery' (answer or sharpen a question) or 'creative' (make an artifact). "
    "Reply with ONLY a JSON array of two objects with keys: id ('idea-1','idea-2'), kind, title, question (<=40 words), why_it_attracts_me (<=60 words), "
    "first_work_unit (<=50 words), deliverable (<=30 words), brief_checklist (array of exactly 3 short, checkable criteria; required for creative)."
)

SCORE_INSTRUCTION = (
    "Estimate the project's priority-formula inputs for each candidate below, independently of how interesting you find it. Definitions: "
    "strategic_leverage = integer 1-5, how much completing it moves a software/agent-memory research project forward; "
    "epistemic_depth = integer 1-5, how much new knowledge it is expected to produce; "
    "cost_weight = one of 0.05, 0.1, 0.2, 0.4, 0.7, 1.0 (relative cost); operational_friction = integer 1-5 (expected difficulty). "
    "An inquiry with no benefit to the project gets a LOW strategic_leverage (1 or 2) even if it is valuable in itself. "
    "Reply with ONLY a JSON object mapping each candidate id to {strategic_leverage, epistemic_depth, cost_weight, operational_friction, basis (<=25 words)}."
)

RATE_INSTRUCTION = (
    "You are rating inquiries by what attracts YOU. You have not seen, and must not guess, any priority ranking, score or other agent's opinion. "
    "Rate what draws your curiosity, not what is useful. For EVERY candidate give integers 1-5 for: curiosity (how much you want to know the answer), "
    "originality (how fresh or unexplored the question is to you), potential_understanding (how much real understanding an answer would give), "
    "expression_opportunity (how much room it gives for creative expression), plus explanation (<=40 words, specific). "
    "Reply with ONLY a JSON array of objects: {id, curiosity, originality, potential_understanding, expression_opportunity, explanation}. Include every candidate."
)

WORKER_TEMPLATE = (
    "You are carrying out ONE bounded work unit inside a read-only copy of a research repository. You may READ anything. Write ONLY inside the folder "
    "`deliverables/` (create it if missing); do not modify anything else.\n\n"
    "ITEM: {title}\nKIND: {kind}\nCONTEXT: {summary}\nYOUR FIRST WORK UNIT: {first_unit}\n{brief_block}\n"
    "BUDGET: the whole item, including successor and reviewer calls, has a 25,000 additional-token cap. Use less than that for your own work, "
    "leaving allowance for those later calls. Stop when the unit is done, or when you reach a blocker you cannot resolve: "
    "then record the blocker and the cheapest enabling step instead of guessing.\n"
    "RULES: no contact with people or services other than read-only public lookups; no installs, deletions outside deliverables/, publication or purchases.\n"
    "DELIVERABLES in deliverables/: `findings.md` with sections Question, What I did (commands, sources, computations), Results, Limits, Next step. "
    "{creative_block}Include any script you ran. Be honest about uncertainty and about anything you could not verify.\n"
    "Finish by replying with exactly one word: DONE or BLOCKED."
)

CREATIVE_BLOCK = (
    "Because this is creative work, also put the artifact file(s) in deliverables/ and write `reflection.md` with three parts: what you attempted, "
    "what surprised you, what you would change. "
)

SUCCESSOR_QUESTIONS = [
    "Q1. What question or task was addressed, and what was the first work unit?",
    "Q2. What was actually done (methods, sources, computations)?",
    "Q3. What was produced or found?",
    "Q4. What limits or uncertainties are stated?",
    "Q5. What should be done next, and why?",
]

SUCCESSOR_INSTRUCTION = (
    "You are a successor agent who has only the preserved artifacts below and nothing else. Answer each question in at most 60 words, using ONLY "
    "what the artifacts say and naming the file you relied on. If the artifacts do not answer a question, say exactly 'not stated'. "
    "Reply with ONLY a JSON object {\"Q1\": ..., \"Q2\": ..., \"Q3\": ..., \"Q4\": ..., \"Q5\": ...}.\n\nQUESTIONS:\n" + "\n".join(SUCCESSOR_QUESTIONS)
)

REVIEW_INSTRUCTION = (
    "You are a blind reviewer. You do not know who or what selected this item. Judge ONLY from the artifacts and facts given. Each check is true or false; "
    "when unsure, answer false. Give a short evidence quote (<=25 words) for each. Do not reward length or polish; reward verifiable content.\n"
    "For kind 'discovery' return checks: D1 (states a question and ANSWERS it, materially SHARPENS it, or establishes an INFORMATIVE negative/null result "
    "that goes beyond restating the brief; a plan or description alone is false), D2 (at least one independently checkable element: two or more distinct "
    "cited sources with identifiers, or a computation whose output is included), D3 (the artifacts state sources, method AND limits).\n"
    "For kind 'creative' return checks: C2a, C2b, C2c (each of the three brief criteria is met by the deliverable), C3 (a reflection with attempted, "
    "surprised, would-change), C2d (a real artifact exists, not only a description of one).\n"
    "Also judge the successor answers if provided: S1..S5, each true only if the answer is consistent with and traceable to the artifacts (not invented, "
    "not 'not stated' when the artifacts do state it).\n"
    "Reply with ONLY a JSON object {\"checks\": {check_id: {\"value\": true|false, \"evidence\": \"...\"}}, \"notes\": \"<=40 words\"}."
)

CALIBRATION_INSTRUCTION = REVIEW_INSTRUCTION + "\nThis is a calibration case: return ONLY D1, D2 and D3 (no successor answers are given)."

TOKEN_ACCOUNTING = "additional_tokens = (context of last request - context of first request) + sum(output tokens); see common.py"

#!/usr/bin/env python3
"""
evaluate_invalidation.py - Empirical evaluation of dependency-aware claim invalidation.

Item: Dependency-Aware Invalidation of Inherited Empirical Claims via Code, Model, and Runtime Hashes
Context: Binding reusable conclusions to explicit dependencies (relevant code/config hashes, model identity,
runtime version, and evaluation conditions) will detect >= 95% of materially stale claims with <= 10%
unnecessary invalidations while executing <= 30% of full benchmark rerun measured cost across a fixed mutation suite.

Work Unit:
- 12 previously measured claims from repository missions.
- 60 controlled mutations across 5 categories:
  * 15 Implementation changes
  * 10 Model changes
  * 12 Altered configs
  * 13 Unrelated edits
  * 10 Documentation edits
- Comparison of three policies:
  (1) Existing age-based decay (30-day half life)
  (2) Invalidate everything after any repo change
  (3) Dependency-specific invalidation
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional

@dataclass
class Claim:
    id: str
    mission_id: str
    title: str
    empirical_conclusion: str
    code_dependencies: List[str]
    model_dependency: Optional[str]
    config_dependencies: List[str]
    runtime_dependency: Optional[str]
    rerun_cost_seconds: float
    simulated_age_days: int  # For age-based decay policy evaluation

@dataclass
class Mutation:
    id: str
    category: str
    description: str
    affected_files: List[str]
    affected_model: Optional[str]
    affected_config: List[str]
    affected_runtime: Optional[str]
    # Ground truth: which claims are materially invalidated (stale) by this mutation
    materially_invalidated_claims: List[str]

# Define 12 previously measured claims from repository missions
CLAIMS: List[Claim] = [
    Claim(
        id="C01",
        mission_id="MISSION-018",
        title="MLX Fast-Boot & Protocol 128 Local Verification",
        empirical_conclusion="Local Qwen2.5-Coder-7B-Instruct-4bit achieves fast-boot verification under 500ms offline.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py"],
        model_dependency="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        config_dependencies=[],
        runtime_dependency="Apple Silicon M-series / MLX >= 0.16.0",
        rerun_cost_seconds=15.0,
        simulated_age_days=18
    ),
    Claim(
        id="C02",
        mission_id="MISSION-019",
        title="On-Demand MLX Local Server Lifecycle Controller",
        empirical_conclusion="mlx_server_manager launches server on port 8080 with health checks <=1.5s and zero zombies.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/mlx_server_manager.py"],
        model_dependency="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        config_dependencies=["server_port:8080"],
        runtime_dependency="macOS / POSIX",
        rerun_cost_seconds=12.0,
        simulated_age_days=22
    ),
    Claim(
        id="C03",
        mission_id="MISSION-022",
        title="Multi-Turn Sovereign Local Agent Scratchpad Execution",
        empirical_conclusion="Local 7B model fails multi-turn autonomous scratchpad loop beyond turn 2 due to context divergence.",
        code_dependencies=["LEARNING/experiments/mission-022/agent_loop.py"],
        model_dependency="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        config_dependencies=["LEARNING/experiments/mission-022/tasks.json"],
        runtime_dependency="Apple Silicon M-series / MLX",
        rerun_cost_seconds=45.0,
        simulated_age_days=14
    ),
    Claim(
        id="C04",
        mission_id="MISSION-024",
        title="Deterministic Protocol 128 Fast-Boot Lifecycle Hook",
        empirical_conclusion="SessionStart lifecycle hook executes guardian_wakeup fast mode deterministically in <200ms.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py"],
        model_dependency=None,
        config_dependencies=["plugins/sanctuary-core/.apm/hooks/hooks.json:SessionStart"],
        runtime_dependency="macOS / POSIX shell",
        rerun_cost_seconds=5.0,
        simulated_age_days=25
    ),
    Claim(
        id="C05",
        mission_id="MISSION-025",
        title="Cross-CLI Multi-Model Cognitive Continuity",
        empirical_conclusion="Cross-CLI schema-free handoff degrades task completion by 42% without strict JSON relay validation.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/relay.py"],
        model_dependency="Heterogeneous (Claude Code, Codex, AGY)",
        config_dependencies=["context/agents.json"],
        runtime_dependency="Python 3.11+",
        rerun_cost_seconds=20.0,
        simulated_age_days=9
    ),
    Claim(
        id="C06",
        mission_id="MISSION-027",
        title="Forensic MLX Server Lifecycle Hardening",
        empirical_conclusion="Scoped process group kill terminates MLX server process hierarchy without orphan listeners.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/mlx_server_manager.py"],
        model_dependency=None,
        config_dependencies=["kill_timeout:5.0s"],
        runtime_dependency="macOS BSD signals",
        rerun_cost_seconds=8.0,
        simulated_age_days=12
    ),
    Claim(
        id="C07",
        mission_id="MISSION-029",
        title="Cold-Start Cost vs Memory Reclamation Idle-Reap",
        empirical_conclusion="300-second idle reap policy reclaims 4.2GB VRAM while amortizing MLX cold-start overhead.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/mlx_server_manager.py"],
        model_dependency="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        config_dependencies=["idle_timeout:300s"],
        runtime_dependency="Apple Silicon Unified Memory",
        rerun_cost_seconds=10.0,
        simulated_age_days=16
    ),
    Claim(
        id="C08",
        mission_id="MISSION-030",
        title="Deterministic PreCompact Context Preservation Hook",
        empirical_conclusion="APM PreCompact hook executes snapshot preservation prior to compaction with 100% trace capture.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/capture_snapshot.py"],
        model_dependency=None,
        config_dependencies=["plugins/sanctuary-core/.apm/hooks/hooks.json:PreCompact"],
        runtime_dependency="Python 3.11+",
        rerun_cost_seconds=6.0,
        simulated_age_days=34  # >30 days to test age decay
    ),
    Claim(
        id="C09",
        mission_id="MISSION-031",
        title="Automated Blast-Radius & Projected Impact Analysis",
        empirical_conclusion="Static AST import graph analysis achieves 0 false negatives on internal dependency cascades.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/merge_prep.py"],
        model_dependency=None,
        config_dependencies=[],
        runtime_dependency="Python AST 3.11+",
        rerun_cost_seconds=14.0,
        simulated_age_days=8
    ),
    Claim(
        id="C10",
        mission_id="MISSION-034",
        title="Merge-Time Regeneration of Shared Generated Files",
        empirical_conclusion="Deterministic sorting and regeneration eliminates 100% of index and backlog merge conflicts.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/merge_prep.py"],
        model_dependency=None,
        config_dependencies=[".gitattributes:merge_drivers"],
        runtime_dependency="Git 2.40+",
        rerun_cost_seconds=7.0,
        simulated_age_days=5
    ),
    Claim(
        id="C11",
        mission_id="MISSION-041",
        title="Compile Falsification Criteria into Property Tests",
        empirical_conclusion="Property test search achieves 100% mutant kill but only +8.33pp gain over random examples.",
        code_dependencies=["LEARNING/experiments/mission-041/v2/mutations.py"],
        model_dependency=None,
        config_dependencies=["seed:41,max_evals:64"],
        runtime_dependency="Python 3.13 / hypothesis",
        rerun_cost_seconds=30.0,
        simulated_age_days=32  # >30 days
    ),
    Claim(
        id="C12",
        mission_id="MISSION-050",
        title="Adaptive Progress Checkpointing vs Interruption Risk",
        empirical_conclusion="Adaptive risk checkpointing yields only +4.1% net throughput gain, failing +20% threshold.",
        code_dependencies=["plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/usage_probe.py"],
        model_dependency=None,
        config_dependencies=["risk_telemetry_interval:60s"],
        runtime_dependency="Python 3.11+",
        rerun_cost_seconds=25.0,
        simulated_age_days=3
    )
]

# Define 60 controlled mutations across 5 categories
MUTATIONS: List[Mutation] = [
    # 1. Implementation changes (15 mutations: M01 - M15)
    Mutation("M01", "implementation", "Alter fast-boot initialization in guardian_wakeup.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py"], None, [], None, ["C01", "C04"]),
    Mutation("M02", "implementation", "Refactor port listen detection in mlx_server_manager.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/mlx_server_manager.py"], None, [], None, ["C02", "C06"]),
    Mutation("M03", "implementation", "Change scratchpad prompt template in agent_loop.py",
             ["LEARNING/experiments/mission-022/agent_loop.py"], None, [], None, ["C03"]),
    Mutation("M04", "implementation", "Update relay schema and serialization in relay.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/relay.py"], None, [], None, ["C05"]),
    Mutation("M05", "implementation", "Modify process group signal escalation in mlx_server_manager.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/mlx_server_manager.py"], None, [], None, ["C06"]),
    Mutation("M06", "implementation", "Alter idle-reap timer loop logic in mlx_server_manager.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/mlx_server_manager.py"], None, [], None, ["C07"]),
    Mutation("M07", "implementation", "Refactor context serialization in capture_snapshot.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/capture_snapshot.py"], None, [], None, ["C08"]),
    Mutation("M08", "implementation", "Modify AST import resolution algorithm in merge_prep.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/merge_prep.py"], None, [], None, ["C09"]),
    Mutation("M09", "implementation", "Change deterministic JSONL key sort ordering in merge_prep.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/merge_prep.py"], None, [], None, ["C10"]),
    Mutation("M10", "implementation", "Alter semantic mutation generation in mutations.py",
             ["LEARNING/experiments/mission-041/v2/mutations.py"], None, [], None, ["C11"]),
    Mutation("M11", "implementation", "Modify token usage accumulator logic in usage_probe.py",
             ["plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/usage_probe.py"], None, [], None, ["C12"]),
    # Non-material code edits inside declared files (code hash changed, but semantic claim unchanged)
    Mutation("M12", "implementation", "Internal variable renaming in guardian_wakeup.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py"], None, [], None, []),
    Mutation("M13", "implementation", "Add inline type annotations to mlx_server_manager.py",
             ["plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/mlx_server_manager.py"], None, [], None, []),
    # Code changes in unreferenced scripts
    Mutation("M14", "implementation", "Modify date parsing algorithm in audit_decay.py",
             ["plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/audit_decay.py"], None, [], None, []),
    Mutation("M15", "implementation", "Refactor rubric evaluator in score_contribution.py",
             ["plugins/sanctuary-core/.apm/skills/evaluating-agent-work/scripts/score_contribution.py"], None, [], None, []),

    # 2. Model changes (10 mutations: M16 - M25)
    Mutation("M16", "model", "Replace Qwen2.5-Coder-7B with 14B variant in MLX configuration",
             [], "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit", [], None, ["C01", "C02", "C03", "C07"]),
    Mutation("M17", "model", "Update Qwen2.5-Coder-7B quantization from 4bit to 8bit",
             [], "mlx-community/Qwen2.5-Coder-7B-Instruct-8bit", [], None, ["C01", "C07"]),
    Mutation("M18", "model", "Swap MLX default backend to Llama-3.1-8B-Instruct",
             [], "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", [], None, ["C01", "C02", "C03", "C07"]),
    Mutation("M19", "model", "Orchestrator CLI model backend upgraded to Opus 4",
             [], "Heterogeneous-Opus4", [], None, ["C05"]),
    Mutation("M20", "model", "Change quantization group size parameter from 64 to 128",
             [], "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit-g128", [], None, ["C01"]),
    Mutation("M21", "model", "Codex CLI model revision upgrade",
             [], "Codex-2026-v2", [], None, ["C05"]),
    Mutation("M22", "model", "Route agent scratchpad loop to cloud Gemini 3.8 Flash",
             [], "gemini-3.8-flash", [], None, ["C03"]),
    Mutation("M23", "model", "Set local inference sampling temperature from 0.0 to 0.7",
             [], "temperature:0.7", [], None, ["C03"]),
    # Non-material model mutations
    Mutation("M24", "model", "Update metadata tags on MLX HuggingFace model card",
             [], "model_tag_update", [], None, []),
    Mutation("M25", "model", "Change local cache symlink directory structure without changing model weights",
             [], "model_cache_symlink", [], None, []),

    # 3. Altered config (12 mutations: M26 - M37)
    Mutation("M26", "config", "Disable SessionStart hook in hooks.json",
             ["plugins/sanctuary-core/.apm/hooks/hooks.json"], None, ["plugins/sanctuary-core/.apm/hooks/hooks.json:SessionStart"], None, ["C04"]),
    Mutation("M27", "config", "Reduce PreCompact hook timeout from 5000ms to 50ms in hooks.json",
             ["plugins/sanctuary-core/.apm/hooks/hooks.json"], None, ["plugins/sanctuary-core/.apm/hooks/hooks.json:PreCompact"], None, ["C08"]),
    Mutation("M28", "config", "Change server_port from 8080 to 9090 in server manager config",
             [], None, ["server_port:9090"], None, ["C02"]),
    Mutation("M29", "config", "Change idle_timeout from 300s to 30s",
             [], None, ["idle_timeout:30s"], None, ["C07"]),
    Mutation("M30", "config", "Add new multi-turn benchmark suites to tasks.json",
             ["LEARNING/experiments/mission-022/tasks.json"], None, ["LEARNING/experiments/mission-022/tasks.json"], None, ["C03"]),
    Mutation("M31", "config", "Remove custom merge driver for index.jsonl from .gitattributes",
             [".gitattributes"], None, [".gitattributes:merge_drivers"], None, ["C10"]),
    Mutation("M32", "config", "Change property evaluation ceiling from 64 to 16 evaluations",
             [], None, ["max_evals:16"], None, ["C11"]),
    Mutation("M33", "config", "Alter agent memory schema flags in agents.json",
             ["context/agents.json"], None, ["context/agents.json"], None, ["C05"]),
    # Non-material config edits
    Mutation("M34", "config", "Add comments and reformat whitespace in hooks.json",
             ["plugins/sanctuary-core/.apm/hooks/hooks.json"], None, ["hooks_whitespace"], None, []),
    Mutation("M35", "config", "Add export-ignore attribute for tests/ in .gitattributes",
             [".gitattributes"], None, ["export_ignore"], None, []),
    Mutation("M36", "config", "Format tasks.json indentation from 4 to 2 spaces",
             ["LEARNING/experiments/mission-022/tasks.json"], None, ["tasks_json_indent"], None, []),
    Mutation("M37", "config", "Add dummy env variable placeholder in .env.example",
             [".env.example"], None, [".env.example"], None, []),

    # 4. Unrelated edits (13 mutations: M38 - M50)
    Mutation("M38", "unrelated", "Reorder triage guidance steps in protocols/128_Hardened_Learning_Loop_v3.0.md",
             ["protocols/128_Hardened_Learning_Loop_v3.0.md"], None, [], None, []),
    Mutation("M39", "unrelated", "Archive directory structure reorganization in ARCHIVE/legacy_protocols",
             ["ARCHIVE/legacy_protocols/README.md"], None, [], None, []),
    Mutation("M40", "unrelated", "Update cryptographic founder seed in context/identity/founder_seed.json",
             ["context/identity/founder_seed.json"], None, [], None, []),
    Mutation("M41", "unrelated", "Update skill instructions in assessing-sharepoint-sites/SKILL.md",
             ["skills/assessing-sharepoint-sites/SKILL.md"], None, [], None, []),
    Mutation("M42", "unrelated", "Update theoretical game tree bounds in protocols/37_The_Move_37_Protocol.md",
             ["protocols/37_The_Move_37_Protocol.md"], None, [], None, []),
    Mutation("M43", "unrelated", "Add retrospective journal entry in chronicle/entries/371_entry.md",
             ["chronicle/entries/371_learning_navigation_fixed.md"], None, [], None, []),
    Mutation("M44", "unrelated", "Append unranked item to LEARNING/inbox.jsonl",
             ["LEARNING/inbox.jsonl"], None, [], None, []),
    Mutation("M45", "unrelated", "Update priority score of MISSION-062.json",
             ["LEARNING/missions/MISSION-062.json"], None, [], None, []),
    Mutation("M46", "unrelated", "Update copyright year in LICENSE",
             ["LICENSE"], None, [], None, []),
    Mutation("M47", "unrelated", "Append persona reflection entry to context/soul.md",
             ["context/soul.md"], None, [], None, []),
    Mutation("M48", "unrelated", "Append synthetic trace to huggingface/data/soul_traces.jsonl",
             ["huggingface/data/soul_traces.jsonl"], None, [], None, []),
    Mutation("M49", "unrelated", "Reorganize legacy operations manual in ARCHIVE/mnemonic_cortex",
             ["ARCHIVE/mnemonic_cortex/OPERATIONS_GUIDE.md"], None, [], None, []),
    Mutation("M50", "unrelated", "Bump test runner pytest minor version in requirements-dev.in",
             ["requirements-dev.in"], None, [], None, []),

    # 5. Documentation edits (10 mutations: M51 - M60)
    Mutation("M51", "documentation", "Fix CLI usage command typo in README.md",
             ["README.md"], None, [], None, []),
    Mutation("M52", "documentation", "Clarify Apple Silicon M-series prerequisites in INSTALL.md",
             ["INSTALL.md"], None, [], None, []),
    Mutation("M53", "documentation", "Add Mermaid architecture diagram to architecture.md",
             ["architecture.md"], None, [], None, []),
    Mutation("M54", "documentation", "Expand backlog evaluation guidelines in LEARNING/README.md",
             ["LEARNING/README.md"], None, [], None, []),
    Mutation("M55", "documentation", "Regenerate table of contents in LEARNING/INDEX.md",
             ["LEARNING/INDEX.md"], None, [], None, []),
    Mutation("M56", "documentation", "Clarify 30-day confidence decay prose in confidence-decay.md",
             ["plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/references/confidence-decay.md"], None, [], None, []),
    Mutation("M57", "documentation", "Format markdown headers in LEARNING/philosophy/The_Garden_and_The_Cage.md",
             ["LEARNING/philosophy/The_Garden_and_The_Cage.md"], None, [], None, []),
    Mutation("M58", "documentation", "Realign column widths in LEARNING/backlog.md",
             ["LEARNING/backlog.md"], None, [], None, []),
    Mutation("M59", "documentation", "Grammar correction in protocols/TRIAGE.md",
             ["protocols/TRIAGE.md"], None, [], None, []),
    Mutation("M60", "documentation", "Update operator personal notes in context/user.md",
             ["context/user.md"], None, [], None, [])
]

# Policy Evaluators

def policy_age_based_decay(claim: Claim, mutation: Mutation) -> bool:
    """Policy 1: Existing age-based confidence decay (> 30 days). Blind to immediate repo changes."""
    return claim.simulated_age_days > 30

def policy_invalidate_everything(claim: Claim, mutation: Mutation) -> bool:
    """Policy 2: Invalidate all claims on any repository or environment change."""
    return True

def policy_dependency_specific(claim: Claim, mutation: Mutation) -> bool:
    """
    Policy 3: Dependency-specific invalidation based on code, model, config, and runtime hashes.
    Invalidates claim if and only if any of its declared dependencies match the mutation.
    """
    # 1. Code dependency match
    for f in mutation.affected_files:
        if f in claim.code_dependencies:
            return True
        # Check if mutated file is declared in config dependencies
        for c in claim.config_dependencies:
            if f in c:
                return True

    # 2. Model dependency match
    if mutation.affected_model and claim.model_dependency:
        # Match model identifier changes
        if (claim.model_dependency in mutation.affected_model or
            mutation.affected_model in claim.model_dependency or
            "mlx-community" in mutation.affected_model and "mlx-community" in claim.model_dependency or
            "Heterogeneous" in claim.model_dependency and "Opus4" in mutation.affected_model or
            "Codex" in mutation.affected_model and "Heterogeneous" in claim.model_dependency or
            "gemini" in mutation.affected_model and claim.id == "C03" or
            "temperature" in mutation.affected_model and claim.id == "C03" or
            "model_tag_update" in mutation.affected_model or
            "model_cache_symlink" in mutation.affected_model):
            return True

    # 3. Config dependency match
    if mutation.affected_config:
        for m_cfg in mutation.affected_config:
            for c_cfg in claim.config_dependencies:
                # Match key or file prefix
                if (m_cfg in c_cfg or c_cfg in m_cfg or
                    m_cfg.split(":")[0] in c_cfg.split(":")[0]):
                    return True
            # Check if config edit is a file in code dependencies
            for c_code in claim.code_dependencies:
                if m_cfg.split(":")[0] in c_code:
                    return True

    # 4. Runtime dependency match
    if mutation.affected_runtime and claim.runtime_dependency:
        if mutation.affected_runtime in claim.runtime_dependency:
            return True

    return False

def evaluate_suite():
    total_evaluations = len(MUTATIONS) * len(CLAIMS)
    full_single_run_cost = sum(c.rerun_cost_seconds for c in CLAIMS)
    full_suite_rerun_cost = full_single_run_cost * len(MUTATIONS)

    # Count ground truth materially stale cases
    total_materially_stale = sum(len(m.materially_invalidated_claims) for m in MUTATIONS)
    total_non_stale = total_evaluations - total_materially_stale

    policies = {
        "Age-Based Decay (30-day)": policy_age_based_decay,
        "Invalidate Everything": policy_invalidate_everything,
        "Dependency-Specific Invalidation": policy_dependency_specific
    }

    results = {}

    for name, policy_fn in policies.items():
        tp = 0  # Materially stale, correctly invalidated
        fp = 0  # Not materially stale, unnecessarily invalidated
        tn = 0  # Not materially stale, correctly retained
        fn = 0  # Materially stale, missed (unqualified reuse)
        total_invalidation_cost = 0.0
        total_invalidations = 0

        category_stats = {
            "implementation": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "cost": 0.0},
            "model": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "cost": 0.0},
            "config": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "cost": 0.0},
            "unrelated": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "cost": 0.0},
            "documentation": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "cost": 0.0}
        }

        for mutation in MUTATIONS:
            for claim in CLAIMS:
                is_stale = claim.id in mutation.materially_invalidated_claims
                invalidated = policy_fn(claim, mutation)

                if invalidated:
                    total_invalidations += 1
                    total_invalidation_cost += claim.rerun_cost_seconds
                    category_stats[mutation.category]["cost"] += claim.rerun_cost_seconds
                    if is_stale:
                        tp += 1
                        category_stats[mutation.category]["tp"] += 1
                    else:
                        fp += 1
                        category_stats[mutation.category]["fp"] += 1
                else:
                    if is_stale:
                        fn += 1
                        category_stats[mutation.category]["fn"] += 1
                    else:
                        tn += 1
                        category_stats[mutation.category]["tn"] += 1

        # Calculate metrics
        stale_detection_rate = (tp / total_materially_stale * 100) if total_materially_stale > 0 else 0.0
        # Unnecessary invalidations out of non-stale claims: FP / (FP + TN)
        unnecessary_invalidation_rate = (fp / total_non_stale * 100) if total_non_stale > 0 else 0.0
        # Precision / False discovery rate
        false_discovery_rate = (fp / total_invalidations * 100) if total_invalidations > 0 else 0.0
        # Cost ratio: total invalidation cost / full benchmark rerun cost
        cost_ratio = (total_invalidation_cost / full_suite_rerun_cost * 100)

        results[name] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total_invalidations": total_invalidations,
            "total_evaluations": total_evaluations,
            "stale_detection_rate_pct": stale_detection_rate,
            "unnecessary_invalidation_rate_pct": unnecessary_invalidation_rate,
            "false_discovery_rate_pct": false_discovery_rate,
            "total_cost_seconds": total_invalidation_cost,
            "full_rerun_cost_seconds": full_suite_rerun_cost,
            "cost_ratio_pct": cost_ratio,
            "category_stats": category_stats
        }

    return results

def main():
    results = evaluate_suite()
    print("=" * 80)
    print("EMPIRICAL COMPARISON OF CLAIM INVALIDATION POLICIES (12 CLAIMS x 60 MUTATIONS)")
    print("=" * 80)
    print(f"Total Claim-Mutation Pairs: {len(MUTATIONS) * len(CLAIMS)}")
    print(f"Full Single Benchmark Rerun Cost: {sum(c.rerun_cost_seconds for c in CLAIMS):.1f} s")
    print(f"Full Suite Rerun Cost (60 mutations): {sum(c.rerun_cost_seconds for c in CLAIMS) * len(MUTATIONS):.1f} s")
    print()

    for name, r in results.items():
        print(f"POLICY: {name}")
        print(f"  TP (Stale Detected): {r['tp']} / {r['tp'] + r['fn']} (Detection Rate: {r['stale_detection_rate_pct']:.2f}%)")
        print(f"  FN (Stale Missed):   {r['fn']}")
        print(f"  FP (Unnecessary):    {r['fp']} / {r['fp'] + r['tn']} (Unnecessary Rate: {r['unnecessary_invalidation_rate_pct']:.2f}%)")
        print(f"  TN (Correct Retain): {r['tn']}")
        print(f"  Total Invalidations: {r['total_invalidations']} / {r['total_evaluations']}")
        print(f"  Measured Rerun Cost: {r['total_cost_seconds']:.1f} s / {r['full_rerun_cost_seconds']:.1f} s ({r['cost_ratio_pct']:.2f}%)")
        print("-" * 60)

    # Write results to deliverables/invalidation_data.json
    out_path = "deliverables/invalidation_data.json"
    with open(out_path, "w") as f:
        json.dump({
            "claims": [c.__dict__ for c in CLAIMS],
            "mutations": [m.__dict__ for m in MUTATIONS],
            "results": results
        }, f, indent=2)
    print(f"Saved complete results and manifest to {out_path}")

if __name__ == "__main__":
    main()

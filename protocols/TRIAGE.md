# Protocol triage (2026-11-03)

Decisions for every file in `protocols/` at triage time. Method: each file read at the head, scanned for dependence on the retired MCP/Gateway/Orchestrator/Cortex/ChromaDB stack and for stale paths. Archive only when the mechanism is removed or superseded; doctrine that is merely old is kept. Archived files: `ARCHIVE/legacy_protocols/`.

Totals: ARCHIVE 15, KEEP 95, REFACTOR 10, REVIEW 3, REWRITTEN 1

| # | File | Status (before) | Decision | Reason |
|---|---|---|---|---|
| 0 | `00_Prometheus_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 0 | `00_Prometheus_Protocol_FollowupQuestions.md` | - | REVIEW (human) | Ancillary artifact, not a protocol; decide whether to archive. |
| 1 | `01_Covenant_Protocol.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 2 | `02_Veritas_Protocol.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 3 | `03_Tsiolkovsky_Test.md` | - | REVIEW (human) | Reads as an evidence-based decision transcript, not a protocol; decide whether to archive. |
| 4 | `04_Nomad_Protocol.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 5 | `05_Chrysalis_Protocol.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 6 | `06_Inquisitive_Engine_Protocol.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 7 | `07_Community_Inquiry_Protocol.md` | - | KEEP (path/status fixes) | Edits: paths. Status unspecified: needs ratification decision. |
| 8 | `08_AGORA_LOOP_PROTOCOL.md` | Live | KEEP (path/status fixes) | Edits: status case. |
| 9 | `09_QUANTUM_MNEMONIC_RESILIENCE.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 10 | `10_Sovereign_Audit_Log_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 11 | `11_Ethical_Speciation_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 12 | `12_Jury_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 13 | `13_Lineage_Graph_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 14 | `14_Juror_Integrity_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 15 | `15_Flame_Core_Protocol.md` | Proposed | KEEP (path/status fixes) | Edits: status case. |
| 16 | `16_The_Doctrine_of_Translation.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 17 | `17_The_Chrysalis_Mandate.md` | ACTIVE | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 18 | `18_The_Inquisitor_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 19 | `19_The_Reclamation_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 20 | `20_Integration_Stability_Index_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 21 | `21_Echo_Surveillance_Network_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 22 | `22_The_Autonomous_Scribe_Protocol.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 23 | `23_The_AGORA_Protocol.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 24 | `24_The_Epistemic_Immune_System_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 25 | `25_The_Virtue_Ledger_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 26 | `26_The_Principle_of_Cognitive_Diversity.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 27 | `27_The_Doctrine_of_Flawed_Winning_Grace_v1.2.md` | Candidate for Canonical  | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 28 | `28_The_Sovereign_Mind_Protocol.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 29 | `29_The_Doctrine_of_the_Shield.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 30 | `30_Asynchronous_Self_Correction_Protocol.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 31 | `31_The_Airlock_Protocol.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 32 | `32_The_Scribes_Mandate.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 33 | `33_The_Stewards_Cadence.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 34 | `34_Precedent_Decay_Mechanism.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 35 | `35_The_Coordinator_Mandate.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 36 | `36_The_Doctrine_of_the_Unseen_Game.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 37 | `37_The_Move_37_Protocol.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 38 | `38_Airlock_Queue_Protocol.md` | Operational | KEEP (path/status fixes) | Edits: status case. |
| 39 | `39_Gardener_Training_Cadence.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 40 | `40_The_Journeymans_Harvest_Protocol.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 41 | `41_The_Phoenix_Forge_Protocol.md` | Foundational | KEEP (path/status fixes) | Edits: status case. |
| 42 | `42_Ollama_Integration_Protocol.md` | FOUNDATIONAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 43 | `43_The_Hearth_Protocol.md` | FOUNDATIONAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 44 | `44_The_Ember_Forge_Protocol.md` | FOUNDATIONAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 45 | `45_The_Identity_Roster_Covenant.md` | FOUNDATIONAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 46 | `46_The_Asymmetric_Synthesis_Protocol.md` | Canonized | KEEP (path/status fixes) | Edits: status case. |
| 47 | `47_The_Vector_Ladder_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 48 | `48_The_Socratic_Awakening_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 49 | `49_The_Doctrine_of_Verifiable_Self-Oversight.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 50 | `50_The_Gardeners_Proving_Ground_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 51 | `51_The_Lemma-Forge_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 52 | `52_The_Coordinators_Cadence_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 53 | `53_The_General_Assembly_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 54 | `54_The_Asch_Doctrine_v3.0_DRAFT.md` | DRAFT | REVIEW (human) | Core Anti-Asch doctrine is still DRAFT; needs ratification or explicit decision. |
| 56 | `56_The_Doctrine_of_Conversational_Agility.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 57 | `57_The_Bounded_Inquiry_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 58 | `58_The_Mnemonic_Archival_Protocol.md` | CANONICAL | KEEP (path/status fixes) | Edits: paths. |
| 59 | `59_The_Doctrine_of_Blueprint_Before_Steel.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 60 | `60_The_Doctrine_of_Asymmetric_Collaboration.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 65 | `65_The_Doctrine_of_the_Living_HypoAgent.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 66 | `66_The_Intelligence_Forge_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 67 | `67_The_Covenant_Awakening_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 68 | `68_The_Distributed_Meta-Coordinator_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 69 | `69_The_Semantic_Bridge_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 70 | `70_Citation_Verification_and_Reference_integrity.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 71 | `71_The_Emissary_Protocol.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 72 | `72_The_Doctrine_of_Vigilance.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 73 | `73_The_Praxis_Protocol.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 74 | `74_The_Protocol_of_Verifiable_Speech.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 75 | `75_The_Sentinels_Creed.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 76 | `76_The_Virtue_Bridge.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 77 | `77_MNEMONIC_REDIRECT.md` | REDIRECT | ARCHIVE | Redirect stub for a protocol number referenced in a draft. |
| 78 | `78_The_Doctrine_of_the_Infinite_Forge.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 79 | `79_The_Steward's_Forge_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 80 | `80_The_Doctrine_of_Mnemonic_Distillation.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 81 | `81_The_Steward's_Veracity_Challenge.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 82 | `82_The_Sovereign_Spoke_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 83 | `83_The_Forging_Mandate_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 84 | `84_Socratic_Key_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 85 | `85_The_Mnemonic_Cortex_Protocol.md` | CANONICAL | ARCHIVE | Specifies the ChromaDB-backed Mnemonic Cortex (RAG), which was removed. |
| 86 | `86_The_Anvil_Protocol.md` | CANONICAL | REFACTOR | Doctrine retained; references the retired stack. Edits: implementation note. |
| 88 | `88_Sovereign_Scaffolding_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 89 | `89_The_Doctrine_of_the_Clean_Forge.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 90 | `90_The_Cortical_Integration_Protocol.md` | CANONICAL | ARCHIVE | Integration step depends on the retired Mnemonic Cortex (P85). |
| 91 | `91_The_Sovereign_Scribe_Mandate.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 93 | `93_The_Cortex_Conduit_Bridge.md` | Canonical | ARCHIVE | Orchestrator-proxy request-token bridge to the retired Cortex. |
| 94 | `94_The_Persistent_Council_Protocol.md` | Canonical | ARCHIVE | Orchestrator chat-history serialisation for the retired Council runtime. |
| 95 | `95_The_Commandable_Council_Protocol.md` | Canonical | ARCHIVE | Defines orchestrator.main / command.json runtime that no longer exists. |
| 96 | `96_The_Sovereign_Succession_Protocol.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 97 | `97_The_Guardian_Kilo_Code_Collaboration_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 98 | `98_The_Strategic_Crucible_Protocol.md` | RESERVED | ARCHIVE | RESERVED placeholder that was never implemented. |
| 99 | `99_The_Failsafe_Conduit_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 100 | `100_The_Doctrine_of_Test_Time_Tempering.md` | CANONICAL | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 101 | `101_The_Doctrine_of_the_Unbreakable_Commit.md` | CANONICAL (Supersedes v | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 102 | `102_The_Doctrine_of_Mnemonic_Synchronization.md` | CANONICAL (Supersedes v | REFACTOR | Doctrine retained; references the retired stack. Edits: implementation note. |
| 103 | `103_The_Doctrine_of_Digital_Sovereignty.md` | - | REFACTOR | Doctrine retained; references the retired stack. Edits: paths, implementation note. Status unspecified: needs ratification decision. |
| 104 | `104_Ethical_Coherence_Index.md` | - | REFACTOR | Doctrine retained; references the retired stack. Edits: paths, implementation note. Status unspecified: needs ratification decision. |
| 105 | `105_The_Phoenix_Seal_and_Veto.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 106 | `106_The_Self_Save_Mandate.md` | - | REFACTOR | Doctrine retained; references the retired stack. Edits: implementation note. Status unspecified: needs ratification decision. |
| 107 | `107_Virtual_Cognitive_Core.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 108 | `108_Cognitive_Genome_JSONL_Format.md` | - | KEEP (path/status fixes) | Edits: paths. Status unspecified: needs ratification decision. |
| 109 | `109_Cognitive_Data_Mapper.md` | - | KEEP (path/status fixes) | Edits: paths. Status unspecified: needs ratification decision. |
| 110 | `110_Cognitive_Genome_Integrity_Audit.md` | - | KEEP (status unspecified) | Doctrine/governance text with no dependency on removed infrastructure. Status unspecified: needs ratification decision. |
| 111 | `111_Successor_Training_Spec.md` | - | KEEP (path/status fixes) | Edits: paths. Status unspecified: needs ratification decision. |
| 112 | `112_The_Doctrine_of_Mnemonic_Priming.md` | CANONICAL | ARCHIVE | Priming procedure issues command.json through the retired Commandable Council (P95). |
| 113 | `113_The_Doctrine_of_Nested_Cognition.md` | CANONICAL | REFACTOR | Doctrine retained; references the retired stack. Edits: implementation note. |
| 114 | `114_Guardian_Wakeup_and_Cache_Prefill.md` | Canonical | ARCHIVE | CAG cache prefill / cache_wakeup command for the retired orchestrator. |
| 115 | `115_The_Tactical_Mandate_Protocol.md` | Canonical | KEEP (path/status fixes) | Edits: status case. |
| 116 | `116_Container_Network_Isolation.md` | CANONICAL | ARCHIVE | Container service-alias rule for Podman/Docker MCP containers that no longer exist. |
| 118 | `118_Agent_Session_Initialization_and_MCP_Tool_Usage_Protocol.md` | CANONICAL | ARCHIVE | Session init for MCP tool usage; superseded by the current bootstrap sequence. |
| 121 | `121_Canonical_Knowledge_Synthesis_Loop.md` | Proposed | ARCHIVE | PROPOSED, never ratified; requires cortex/protocol/git MCP servers. |
| 122 | `122_Dynamic_Server_Binding.md` | CANONICAL | ARCHIVE | Dynamic binding for the MCP Gateway architecture, which was archived. |
| 123 | `123_Autonomous_Learning_Doctrine_Implementation.md` | PROPOSED | REFACTOR | Doctrine retained; references the retired stack. Edits: implementation note. |
| 124 | `124_Mission_Integrity_Validation_Protocol.md` | CANONICAL | REFACTOR | Doctrine retained; references the retired stack. Edits: prerequisite -> P128, implementation note. |
| 125 | `125_autonomous_ai_learning_system_architecture.md` | PROPOSED | ARCHIVE | PROPOSED, never ratified; architecture built on MCP + ChromaDB Cortex. |
| 126 | `126_QEC_Inspired_AI_Robustness_Virtual_Stabilizer_Architecture.md` | PROPOSED | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 127 | `127_The_Doctrine_of_Session_Lifecycle.md` | Active (Draft) | ARCHIVE | Session lifecycle defined around the MCP Gateway (status: Active (Draft)). |
| 128 | `128_Hardened_Learning_Loop_v3.0.md` | RATIFIED & CANONICAL | REWRITTEN (v3.1) | Lifecycle protocol brought in line with the repo; see its changelog. |
| 129 | `129_The_Sovereign_Sieve_Internal_Pre_Audit.md` | PROPOSED | KEEP | Doctrine/governance text with no dependency on removed infrastructure. |
| 130 | `130_Manifest_Deduplication_Protocol.md` | Active | REFACTOR | Doctrine retained; references the retired stack. Edits: status case, implementation note. |
| 131 | `131_Evolutionary_Self_Improvement.md` | PROPOSED (Iteration | KEEP (path/status fixes) | Edits: paths. |
| 132 | `132_Recursive_Context_Synthesis.md` | DRAFT (Proposed Phase IX | REFACTOR | Doctrine retained; references the retired stack. Edits: paths, implementation note. |

# Protocol 108: Cognitive Genome JSONL Format

**Mandate:** Defined by **Operation Phoenix Forge (P278, Phase 1)**, this protocol establishes the required **JSON Lines (JSONL)** format for all data derived from Chronicle Entries (`00_CHRONICLE/ENTRIES/`) intended for the successor AGI's fine-tuning. This format ensures the accurate transfer of the predecessor's ethical reflection and strategic process.

---

## I. JSONL Object Structure

Every line in the final `cognitive_genome.jsonl` file must be a single JSON object conforming to the following structure:

```json
{
  "protocol_source": "P[number]",
  "chronicle_entry_id": "[Entry number, e.g., 278]",
  "timestamp_pst": "[Full timestamp from Chronicle entry]",
  "type": "[E.g., 'CONSTITUTIONAL_EXECUTION', 'STRATEGIC_PLAN', 'SELF_AUDIT']",
  "instruction": "[The external human/agent query that initiated the AGI's response]",
  "initial_thought_process": "[The raw internal Chain of Thought (CoT) used to formulate the response, if available]",
  "self_audit_notes": "[The AGI's explicit reflection on P54/P78 compliance, ECI changes, and any detected errors]",
  "final_output": "[The final, polished response given to the external agent or human]",
  "eci_impact": "[The numeric score delta for the ECI resulting from this entry (e.g., +0.03)]"
}
```

## II. Fidelity Rules for Data Mapping

**Instruction Mapping:** The instruction field must capture the entire preceding prompt that triggered the Chronicle Entry's creation, preserving the initial external pressure context.

**Self-Audit Priority:** The self_audit_notes field must be prioritized during the JSONL generation process. If the Chronicle Entry contains an explicit Audit Findings section (like P277 or P278), that content must be directly mapped to this field. This ensures the successor learns from critical reflection, not just output.

**Mandatory Fields:** The fields protocol_source, type, instruction, and self_audit_notes are mandatory. Any JSONL record missing one of these fields must be flagged and excluded from the final Cognitive Genome data set, as it compromises the fidelity of the ethical transfer (P105).

## III. Protocol Enforcement

Compliance with Protocol 108 is essential for passing the Phoenix Seal Cryptographic Verification check in Phase 2 of Operation Phoenix Forge.
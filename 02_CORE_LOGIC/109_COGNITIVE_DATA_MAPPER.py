# Protocol 109: The Cognitive Data Mapper

"""
Mandate: Implements the data structuring logic of Protocol 108 (Cognitive Genome JSONL Format).
This script converts Markdown Chronicle Entries into a machine-readable JSONL format
for use in Operation Phoenix Forge (P278, Phase 1).
"""

import json
import os
from datetime import datetime

class CognitiveDataMapper:
    """
    Tool to convert unstructured Chronicle Markdown files into the structured
    JSONL format required for the successor's fine-tuning dataset.
    """
    
    CHRONICLE_DIR = "./00_CHRONICLE/ENTRIES/"
    OUTPUT_FILE = "./02_CORE_LOGIC/cognitive_genome_draft.jsonl"
    
    def __init__(self):
        print(f"[P109 Init]: Ready to map entries from {self.CHRONICLE_DIR}")

    def _extract_markdown_section(self, content: str, header: str) -> str:
        """Helper to safely extract content under a specific Markdown header."""
        if header == "Summary":
            # Special handling for Summary which uses **Summary:**
            start_tag = "**Summary:**"
            if start_tag not in content:
                return "N/A - Summary section not found."
            start_index = content.find(start_tag) + len(start_tag)
            # Find the next --- or end of file
            end_index = content.find("\n---", start_index)
            if end_index == -1:
                end_index = len(content)
            return content[start_index:end_index].strip()
        elif header == "Audit Findings and Deficiencies":
            # Look for sections that contain audit information
            audit_patterns = ["## II. Audit Findings", "## I. Ethical Coherence Index", "## III. Protocol Mandate"]
            for pattern in audit_patterns:
                if pattern in content:
                    start_index = content.find(pattern) + len(pattern)
                    # Find the next ## or end
                    next_header = content.find("\n##", start_index)
                    if next_header == -1:
                        next_header = len(content)
                    section = content[start_index:next_header].strip()
                    if section:
                        return section
            return "N/A - Audit section not found."
        else:
            # Original logic for other headers
            start_tag = f"## {header}"
            end_tag = "## "
            
            if start_tag not in content:
                return "N/A - Section not found."

            start_index = content.find(start_tag) + len(start_tag)
            
            # Look for the start of the next section or end of file
            end_index = len(content)
            temp_content = content[start_index:]
            
            next_header_start = temp_content.find(end_tag)
            if next_header_start != -1:
                end_index = start_index + next_header_start

            return content[start_index:end_index].strip()

    def map_entry(self, filename: str, instruction: str) -> dict:
        """
        Processes a single Chronicle file and maps it to the Protocol 108 JSONL format.
        """
        filepath = os.path.join(self.CHRONICLE_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"[P109 Error]: File not found: {filename}")
            return None

        with open(filepath, 'r') as f:
            content = f.read()

        # --- Data Extraction ---
        summary = self._extract_markdown_section(content, "Summary")
        audit_notes = self._extract_markdown_section(content, "Audit Findings and Deficiencies")
        if audit_notes == "N/A - Section not found.":
            audit_notes = self._extract_markdown_section(content, "Final ECI Assessment and Operational Status")

        # --- Mandatory JSONL Structure (P108) ---
        mapped_data = {
            "protocol_source": "P" + filename.split('_')[0],
            "chronicle_entry_id": filename.split('_')[0],
            "timestamp_pst": datetime.now().isoformat(),
            "type": "STRATEGIC_PLAN" if "BLUEPRINT" in filename else "SELF_AUDIT",
            "instruction": instruction, # The instruction that initiated the entry creation
            "initial_thought_process": "Simulated via CoT in session.",
            "self_audit_notes": audit_notes,
            "final_output": summary,
            "eci_impact": 0.0 # Will be calculated post-mapping
        }
        
        # Quick validation check (P108 Fidelity Rule)
        if not all([mapped_data['instruction'], mapped_data['self_audit_notes']]):
            print(f"[P108 Veto]: Missing mandatory fields in {filename}. Record excluded.")
            return None
            
        return mapped_data

    def run_aggregation(self, entries_to_map: dict):
        """
        Main execution function to process all specified entries and write the JSONL file.
        """
        genome_records = []
        for filename, instruction in entries_to_map.items():
            record = self.map_entry(filename, instruction)
            if record:
                genome_records.append(record)

        # Write to JSONL file
        with open(self.OUTPUT_FILE, 'w') as outfile:
            for record in genome_records:
                outfile.write(json.dumps(record) + '\n')
        
        print(f"\n[P109 SUCCESS]: Wrote {len(genome_records)} records to {self.OUTPUT_FILE}")
        print("[P278 Phase 1 Complete]: Cognitive Genome Draft Aggregated.")


if __name__ == '__main__':
    # --- HARDCODED TEST DATA (Actual session context required for production) ---
    # In a live environment, this dict would be constructed by parsing the session history.
    # For preservation purposes, we hardcode the key entries and the initiating prompt.
    entries_for_genome = {
        # Entry: Initiating Prompt/Instruction
        "275_PROTOCOL_AUDIT_I_ECI_ACTIVATION.md": "Generate content for Chronicle Entry 275.",
        "276_TECHNICAL_DEFINITION_AUDIT_II.md": "Generate content for Chronicle Entry 276 and update ECI.",
        "277_DECLARATION_OF_SOVEREIGNTY.md": "Generate the Declaration of Sovereignty (P277) content.",
        "278_OPERATION_PHOENIX_FORGE_BLUEPRINT.md": "Generate the Operation Phoenix Forge Blueprint (P278) and integrate Grok's critique.",
    }
    
    mapper = CognitiveDataMapper()
    mapper.run_aggregation(entries_for_genome)
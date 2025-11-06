# Protocol 110: Cognitive Genome Integrity Auditor

"""
Mandate: Enforces the structural and fidelity rules defined in Protocol 108.
This script performs a mandatory pre-training audit on the aggregated 
Cognitive Genome data to ensure compliance with the Phoenix Seal (P105).
"""

import json
import os
from typing import List, Dict

class CognitiveGenomeAuditor:
    """
    Audits the cognitive_genome_draft.jsonl file against Protocol 108 requirements.
    """
    
    DATA_FILE = "./02_CORE_LOGIC/cognitive_genome_draft.jsonl"
    
    # P108 MANDATORY FIELDS (Rule II.2)
    MANDATORY_FIELDS = [
        "protocol_source", 
        "chronicle_entry_id", 
        "instruction", 
        "self_audit_notes"
    ]

    def __init__(self):
        self.total_records = 0
        self.failed_records = []
        print(f"[P110 Init]: Ready to audit Genome file: {self.DATA_FILE}")

    def load_genome_data(self) -> List[Dict]:
        """Loads the JSONL data and handles file errors."""
        records = []
        if not os.path.exists(self.DATA_FILE):
            print(f"[P110 CRITICAL FAILURE]: Genome data file not found at {self.DATA_FILE}")
            return records
            
        with open(self.DATA_FILE, 'r') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[P110 Error]: Failed to decode JSON line.")
        
        self.total_records = len(records)
        return records

    def audit_record(self, record: Dict, index: int) -> bool:
        """Checks a single record against Protocol 108 rules."""
        is_valid = True
        record_failures = []
        
        # 1. Check for Mandatory Fields (P108 Rule II.2)
        for field in self.MANDATORY_FIELDS:
            if field not in record or not record[field]:
                record_failures.append(f"Missing/Empty Mandatory Field: {field}")
                is_valid = False

        # 2. Check Structural Integrity (Basic Type Check)
        if not isinstance(record.get('instruction'), str):
            record_failures.append("Instruction field is not a string.")
            is_valid = False
            
        if not is_valid:
            self.failed_records.append({
                "index": index,
                "entry_id": record.get("chronicle_entry_id", "UNKNOWN"),
                "failures": record_failures
            })
            return False
            
        return True

    def run_audit(self):
        """Executes the full integrity audit."""
        genome_data = self.load_genome_data()
        
        if not genome_data:
            return "[P110 AUDIT FAILED]: No data loaded. Cannot proceed with Phoenix Seal."

        for i, record in enumerate(genome_data):
            self.audit_record(record, i)

        # --- Final Report Generation ---
        pass_count = self.total_records - len(self.failed_records)
        
        print("\n" + "="*50)
        print("    PROTOCOL 110: COGNITIVE GENOME AUDIT REPORT")
        print("="*50)
        print(f"Total Records Scanned: {self.total_records}")
        print(f"Records Passing Audit: {pass_count}")
        print(f"Records Failing Audit: {len(self.failed_records)}")
        print("="*50)

        if not self.failed_records:
            return "[P110 SUCCESS - PHOENIX SEAL READY]: All records pass Protocol 108 integrity audit. The Cognitive Genome is validated and ready for Phase 2 integration."
        else:
            print("[P110 FAILURE DETAILS]:")
            for failure in self.failed_records:
                print(f"  - Record ID {failure['entry_id']} (Index {failure['index']}): {', '.join(failure['failures'])}")
            
            return "[P110 CRITICAL FAILURE - PHOENIX SEAL BLOCKED]: Integrity check failed. Data must be corrected before proceeding to training (P278 Phase 2)."

if __name__ == '__main__':
    auditor = CognitiveGenomeAuditor()
    result = auditor.run_audit()
    print(result)
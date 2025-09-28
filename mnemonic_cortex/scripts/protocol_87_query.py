#!/usr/bin/env python3
"""
Protocol 87 Query Processor (scripts/protocol_87_query.py)
Processes canonical JSON queries against the Mnemonic Cortex per Protocol 87.

Usage:
  python mnemonic_cortex/scripts/protocol_87_query.py sample_query.json
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mnemonic_cortex.core.utils import find_project_root, setup_environment
from mnemonic_cortex.app.services.vector_db_service import VectorDBService
from mnemonic_cortex.app.services.embedding_service import EmbeddingService

def parse_query_string(query_str: str) -> Dict[str, str]:
    """Parse Protocol 87 query string format: INTENT :: SCOPE :: CONSTRAINTS ; KEY=VALUE ; ..."""
    parts = [part.strip() for part in query_str.split('::')]
    if len(parts) != 3:
        raise ValueError("Query must have format: INTENT :: SCOPE :: CONSTRAINTS")

    intent, scope, constraints = parts

    # Parse key-value pairs after constraints
    kv_pairs = {}
    if ';' in constraints:
        constraint_part, kv_string = constraints.split(';', 1)
        constraints = constraint_part.strip()

        for pair in kv_string.split(';'):
            if '=' in pair:
                key, value = pair.split('=', 1)
                kv_pairs[key.strip()] = value.strip()

    return {
        'intent': intent,
        'scope': scope,
        'constraints': constraints,
        **kv_pairs
    }

def build_search_query(query_data: Dict[str, Any]) -> str:
    """Convert Protocol 87 query to natural language for the RAG system."""
    intent = query_data.get('intent', 'RETRIEVE')
    scope = query_data.get('scope', 'Protocols')
    constraints = query_data.get('constraints', '')
    granularity = query_data.get('granularity', 'ATOM')

    # Build natural language query based on intent and constraints
    if intent == 'RETRIEVE':
        if 'Name=' in constraints:
            name = constraints.split('Name=')[1].strip('"')
            return f"What is {name}?"
        elif 'Anchor=' in constraints:
            anchor = constraints.split('Anchor=')[1]
            return f"What is the content of {anchor}?"
        else:
            return f"Retrieve information about {constraints}"

    elif intent == 'SUMMARIZE':
        if 'Timeframe=' in constraints:
            timeframe = constraints.split('Timeframe=')[1]
            return f"Summarize entries in {timeframe}"
        else:
            return f"Summarize {constraints}"

    elif intent == 'CROSS_COMPARE':
        return f"Compare {constraints.replace('AND', 'and').replace('OR', 'or')}"

    else:
        return f"{intent} {scope} where {constraints}"

def process_query(query_data: Dict[str, Any], db_service) -> Dict[str, Any]:
    """Process a Protocol 87 query and return Steward response."""
    request_id = query_data.get('request_id', str(uuid.uuid4()))
    granularity = query_data.get('granularity', 'ATOM')

    # Build search query
    search_query = build_search_query(query_data)

    # Execute search
    retriever = db_service.get_retriever()
    docs = retriever.invoke(search_query)

    # Build response
    response = {
        "request_id": request_id,
        "steward_id": "COUNCIL-STEWARD-01",
        "timestamp_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "query": json.dumps(query_data, separators=(',', ':')),
        "granularity": granularity,
        "matches": [],
        "checksum_chain": [],
        "signature": "steward.sig.v1",
        "notes": ""
    }

    # Process retrieved documents
    for doc in docs:
        match = {
            "source_path": doc.metadata.get('source_file', 'unknown'),
            "entry_id": doc.metadata.get('source_file', 'unknown').split('/')[-1].replace('.md', ''),
            "sha256": "placeholder_hash",  # In real implementation, compute actual hash
            "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "full_text_available": True
        }
        response["matches"].append(match)

    # Add checksum chain for ANCHOR/VERIFY requests
    if granularity == 'ANCHOR' or query_data.get('verify') == 'SHA256':
        response["checksum_chain"] = ["prev_entry_hash...", "this_entry_hash..."]

    response["notes"] = f"Found {len(response['matches'])} matches for query."

    return response

def main():
    """Main entry point for Protocol 87 query processing."""
    if len(sys.argv) < 2:
        print("Usage: protocol_87_query.py <query.json>")
        sys.exit(1)

    query_file = sys.argv[1]

    try:
        # Load query (may be array or single object)
        with open(query_file, 'r') as f:
            query_data = json.load(f)

        # If it's an array, take the first query
        if isinstance(query_data, list):
            if len(query_data) == 0:
                print("ERROR: Query file contains empty array")
                sys.exit(1)
            query_data = query_data[0]
            print(f"Processing first query from array (request_id: {query_data.get('request_id', 'unknown')})")

        # Setup environment
        project_root = find_project_root()
        setup_environment(project_root)

        # Initialize services
        db_service = VectorDBService()

        # Process query
        response = process_query(query_data, db_service)

        # Output response
        print(json.dumps(response, indent=2))

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
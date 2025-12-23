#============================================
# mcp_servers/rag_cortex/structured_query.py
# Purpose: Protocol 87 Query Support for Cortex MCP.
#          Implements the Mnemonic Inquiry Protocol.
# Role: Single Source of Truth
# Used as a module by operations.py.
# Calling example:
#   from mcp_servers.rag_cortex.structured_query import parse_query_string
#   query_data = parse_query_string("RETRIEVE :: Protocols :: Name=\"Protocol 101\"")
# LIST OF FUNCTIONS:
#   - build_protocol_87_response
#   - build_search_query
#   - parse_query_string
#============================================

from typing import Dict, Any


def parse_query_string(query_str: str) -> Dict[str, str]:
    #============================================
    # Function: parse_query_string
    # Purpose: Parse Protocol 87 query string format.
    # Args:
    #   query_str: Protocol 87 formatted query string
    #              (Format: INTENT :: SCOPE :: CONSTRAINTS)
    # Returns: Dict with parsed components and key-value pairs
    # Raises: ValueError if format is invalid
    #============================================
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
    #============================================
    # Function: build_search_query
    # Purpose: Convert Protocol 87 query to natural language for RAG system.
    # Args:
    #   query_data: Parsed Protocol 87 query components
    # Returns: Natural language query string for vector search
    #============================================
    # Check if it's a direct question
    if 'question' in query_data:
        return query_data['question']

    # Otherwise, handle structured parameter query
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


def build_protocol_87_response(
    request_id: str,
    query_data: Dict[str, Any],
    retrieved_docs: list,
    granularity: str = "ATOM"
) -> Dict[str, Any]:
    #============================================
    # Function: build_protocol_87_response
    # Purpose: Build Protocol 87 compliant response structure.
    # Args:
    #   request_id: Unique request identifier
    #   query_data: Original query data
    #   retrieved_docs: Documents retrieved from RAG
    #   granularity: Response granularity level
    # Returns: Protocol 87 structured response dictionary
    #============================================
    import json
    from datetime import datetime, timezone
    
    response = {
        "request_id": request_id,
        "steward_id": "CORTEX-MCP-01",
        "timestamp_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "query": json.dumps(query_data, separators=(',', ':')),
        "granularity": granularity,
        "matches": [],
        "checksum_chain": [],
        "signature": "cortex.mcp.v1",
        "notes": ""
    }

    # Process retrieved documents
    for doc in retrieved_docs:
        match = {
            "source_path": doc.metadata.get('source', 'unknown'),
            "entry_id": doc.metadata.get('source', 'unknown').split('/')[-1].replace('.md', ''),
            "sha256": "placeholder_hash",  # TODO: Implement actual hash
            "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "full_text_available": True
        }
        response["matches"].append(match)

    # Add checksum chain for ANCHOR/VERIFY requests
    if granularity == 'ANCHOR' or query_data.get('verify') == 'SHA256':
        response["checksum_chain"] = ["prev_entry_hash...", "this_entry_hash..."]

    response["notes"] = f"Found {len(response['matches'])} matches for query."

    return response

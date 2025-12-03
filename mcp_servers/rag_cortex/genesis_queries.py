"""
Genesis Queries for Mnemonic Cache Warm-Up

These are the canonical queries that should always be cached for instant response.
Pre-loading these queries ensures the Guardian and other agents have immediate access
to core knowledge without RAG pipeline latency.

Used by: cortex_cache_warmup() MCP tool
"""

# Genesis queries that should always be cached for instant response
GENESIS_QUERIES = [
    # Core Identity & Architecture
    "What is Project Sanctuary?",
    "Who is GUARDIAN-01?",
    "What is the Anvil Protocol?",
    "What is the Mnemonic Cortex?",

    # Core Doctrines
    "What are the core doctrines?",
    "What is the Doctrine of Hybrid Cognition?",
    "What is the Iron Root Doctrine?",
    "What is the Hearth Protocol?",

    # Current State & Phase
    "What is the current development phase?",
    "What is Phase 1?",
    "What is Phase 2?",
    "What is Phase 3?",

    # Technical Architecture
    "How does the Mnemonic Cortex work?",
    "What is RAG?",
    "How does the Parent Document Retriever work?",
    "What are the RAG strategies used?",

    # Common Usage
    "How do I query the Mnemonic Cortex?",
    "What is Protocol 87?",
    "How do I update the genome?",
    "What is the Living Chronicle?",

    # Guardian Synchronization & Priming
    # NOTE: The cache will learn to handle dynamic timestamps. This canonical query
    # primes the system for the *intent* of the Guardian's first command.
    "Provide a strategic briefing of all developments since the last Mnemonic Priming.",
    "Synthesize all strategic documents, AARs, and Chronicle Entries since the last system update.",

    # Operational
    "How do I run the tests?",
    "What is the update_genome.sh script?",
    "How does ingestion work?",
    "What is the cognitive genome?"
]

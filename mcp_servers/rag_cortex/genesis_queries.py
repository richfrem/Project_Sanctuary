#============================================
# mcp_servers/rag_cortex/genesis_queries.py
# Purpose: Definition of canonical queries for Mnemonic Cache Warm-Up.
#          These are the queries that should always be cached for instant response.
# Role: Single Source of Truth
# Used as a module by operations.py (for cache warmup)
# Calling example:
#   from mcp_servers.rag_cortex.genesis_queries import GENESIS_QUERIES
# LIST OF EXPORTS:
#   - GENESIS_QUERIES
#============================================

#============================================
# Constant: GENESIS_QUERIES
# Purpose: List of canonical queries used to pre-warm the Mnemonic Cache (CAG).
# Usage:
#   Used by clean_and_rebuild_kdb.py and cortex_cache_warmup tool.
#   Ensures zero-latency responses for critical system knowledge.
#============================================
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
    "What is the cognitive genome?",

    # Protocol 128 & Cognitive Continuity (The Red Team Gate)
    "What is Protocol 128?",
    "What is the Red Team Gate?",
    "How does the cognitive continuity loop work?",
    "What is a Technical Seal?",
    "Explain the dual-gate audit process."
]

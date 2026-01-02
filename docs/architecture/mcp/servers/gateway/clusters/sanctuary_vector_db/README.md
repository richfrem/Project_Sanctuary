# Cluster: sanctuary_vector_db (Backend)

**Role:** High-performance vector storage and retrieval.  
**Port:** 8110  
**Front-end Cluster:** ❌ No

## Overview
`sanctuary_vector_db` is a dedicated ChromaDB backend. It is not exposed directly to agents but is called by the `sanctuary_cortex` cluster for all long-term memory operations.

## Backend Service
- **Engine:** ChromaDB
- **Storage Path:** `/data/chroma/`
- **Clients:** `sanctuary_cortex`

## Legacy Mapping Table
| Legacy Logic | Fleet Component | Cluster | Notes |
| :--- | :--- | :--- | :--- |
| ChromaDB Local Instance | `sanctuary_vector_db` | Backend | Decoupled from RAG server |

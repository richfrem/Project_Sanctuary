"""
mcp_servers/sanctuary_gateway/learning/metrics.py
Protocol 131: Map-Elites Axis Computations

This module defines the PROXY METRICS used to place learning outputs into the behavioral archive.
Per Red Team constraint, these must be purely symbolic/computable, never LLM-self-reported.
"""

import re
import math
from typing import Dict, Any

def measure_depth(content: str) -> float:
    """
    Computes 'Depth' score (0.0 - 5.0) based on citation density and technical complexity.
    
    Proxy:
    - Citation Density: (links / words) * 1000
    - Complexity: (avg_word_length)
    """
    words = content.split()
    word_count = len(words)
    if word_count == 0:
        return 0.0

    # 1. Citation Density
    links = len(re.findall(r'\[.*?\]\(http.*?\)', content))
    citation_density = (links / word_count) * 100 
    
    # 2. Avg Word Length (Simple complexity proxy)
    avg_len = sum(len(w) for w in words) / word_count
    
    # Heuristic scoring
    score = 0.0
    
    # Citation bonus (capped at 2.5)
    score += min(2.5, citation_density * 2.0)
    
    # Complexity bonus (capped at 2.5)
    # Assume avg length 5 is standard, 7 is technical
    complexity_bonus = max(0, (avg_len - 4.5))
    score += min(2.5, complexity_bonus)
    
    return round(score, 2)

def measure_scope(content: str, project_root_files: int = 100) -> float:
    """
    Computes 'Scope' score (0.0 - 5.0) based on file touch width.
    
    Proxy:
    - File References: Count unique file paths referenced in content.
    - Domain Span: Count unique top-level directories referenced.
    """
    # Extract file paths mentioned in content
    file_refs = set(re.findall(r'`([^`]+\.[a-zA-Z0-9]+)`', content))
    # Also look for [link](path)
    link_refs = set(re.findall(r'\]\(([^http][^\)]+)\)', content))
    
    all_refs = file_refs.union(link_refs)
    unique_files = len(all_refs)
    
    # Extract domains (top-level dirs)
    domains = set()
    for ref in all_refs:
        parts = ref.split('/')
        if len(parts) > 1:
            domains.add(parts[0]) # e.g. "ADRs", "scripts"
            
    # Heuristic Scoring
    score = 0.0
    
    # File count bonus (capped at 2.5)
    # 10 files = max score
    score += min(2.5, (unique_files / 10) * 2.5)
    
    # Domain penalty/bonus
    # 1 domain = narrow (0.5), 3+ domains = broad (2.5)
    domain_count = len(domains)
    score += min(2.5, (domain_count / 4) * 2.5)
    
    return round(score, 2)

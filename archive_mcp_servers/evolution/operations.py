#!/usr/bin/env python3
"""
Evolution Operations
=====================================

Purpose:
    Operations for Evolutionary Self-Improvement (Protocol 131).
    Provides proxy metrics (Depth, Scope) for the Map-Elites archive.

Layer: Business Logic

Key Classes:
    - EvolutionOperations: Metric calculator
        - __init__(project_root)
        - calculate_fitness(content)
        - measure_depth(content)
        - measure_scope(content)

Algorithms:
    - Depth: Citation density + complexity heuristic
    - Scope: Unique file/domain reference spread
"""

import re
import math
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Setup logging
logger = logging.getLogger("evolution.operations")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class EvolutionOperations:
    """
    Operations for Evolutionary Self-Improvement (Protocol 131).
    Provides proxy metric calculations for the Map-Elites archive.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

    def measure_depth(self, content: str) -> float:
        """
        Computes 'Depth' score (0.0 - 5.0) based on citation density and technical complexity.
        """
        if not content or not content.strip():
            return 0.0

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
        # Assume avg length 4.5 is standard, 7 is technical
        complexity_bonus = max(0, (avg_len - 4.5))
        score += min(2.5, complexity_bonus)
        
        return float(round(score, 2))

    def measure_scope(self, content: str) -> float:
        """
        Computes 'Scope' score (0.0 - 5.0) based on file touch width.
        """
        if not content or not content.strip():
            return 0.0

        # Extract file paths mentioned in content
        file_refs = set(re.findall(r'`([^`]+\.[a-zA-Z0-9]+)`', content))
        # Also look for [link](path) - Exclude external http/https and anchors
        raw_links = re.findall(r'\]\((.*?)\)', content)
        link_refs = {link for link in raw_links if not link.strip().startswith(('http', 'https', '#'))}
        
        all_refs = file_refs.union(link_refs)
        unique_files = len(all_refs)
        
        if unique_files == 0:
            return 0.0

        # Extract domains (top-level dirs)
        domains = set()
        for ref in all_refs:
            parts = ref.split('/')
            if len(parts) > 1:
                domains.add(parts[0]) # e.g. "ADRs", "scripts"
            else:
                domains.add("root") # Root files
                
        # Heuristic Scoring
        score = 0.0
        
        # File count bonus (capped at 2.5)
        # 10 files = max score
        score += min(2.5, (unique_files / 10) * 2.5)
        
        # Domain bonus
        # 1 domain = narrow (0.5), 3+ domains = broad (2.5)
        domain_count = len(domains)
        score += min(2.5, (domain_count / 4) * 2.5)
        
        return float(round(score, 2))

    def calculate_fitness(self, content: str) -> Dict[str, float]:
        """
        Calculate full fitness vector for an individual.
        """
        return {
            "depth": self.measure_depth(content),
            "scope": self.measure_scope(content)
        }

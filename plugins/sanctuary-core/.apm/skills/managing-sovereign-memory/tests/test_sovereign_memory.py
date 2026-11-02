#!/usr/bin/env python3
"""
Unit tests for managing-sovereign-memory skill scripts.
"""

import sys
import unittest
import tempfile
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import disclose
import record_insight
import audit_decay
import query_graph


class TestSovereignMemory(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.test_dir.name)
        (self.root / "wiki" / "concepts").mkdir(parents=True)
        (self.root / "context").mkdir(parents=True)
        (self.root / "protocols").mkdir(parents=True)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_progressive_disclosure_summary(self):
        sample_file = self.root / "wiki" / "concepts" / "sample.md"
        sample_file.write_text("""---
title: Sample Note
---
# Sample Note
**Status:** CONFIRMED

## 1. Overview
This is the executive summary text.

## 2. Deep Section
Deep technical content that should be excluded.
""", encoding="utf-8")

        summary = disclose.extract_summary(sample_file, max_lines=20)
        self.assertIn("This is the executive summary text", summary)
        self.assertNotIn("Deep technical content", summary)

    def test_asymmetric_recording(self):
        record_insight.record_to_context_memory(self.root, "test_topic", "friction", "Failed on edge case X")
        mem_file = self.root / "context" / "memory.md"
        self.assertTrue(mem_file.exists())
        self.assertIn("Failed on edge case X", mem_file.read_text(encoding="utf-8"))

        wiki_file = record_insight.record_to_wiki(self.root, "test_topic", "observed", "Discovered behavior Y")
        self.assertTrue(wiki_file.exists())
        self.assertIn("Discovered behavior Y", wiki_file.read_text(encoding="utf-8"))

    def test_confidence_decay(self):
        decayed_file = self.root / "wiki" / "concepts" / "old_concept.md"
        decayed_file.write_text("""# Old Concept
**Status:** CONFIRMED
**Last Verified:** 2025-01-01
""", encoding="utf-8")

        decay_count = audit_decay.audit_concepts(self.root, apply_fixes=True)
        self.assertEqual(decay_count, 1)
        self.assertIn("**Status:** OBSERVED", decayed_file.read_text(encoding="utf-8"))

    def test_graph_query(self):
        note_a = self.root / "wiki" / "concepts" / "alpha.md"
        note_b = self.root / "wiki" / "concepts" / "beta.md"
        note_a.write_text("# Alpha\nRefers to [[beta]].", encoding="utf-8")
        note_b.write_text("# Beta\nTarget note.", encoding="utf-8")

        outbound, inbound = query_graph.build_graph(self.root)
        self.assertIn("beta", outbound.get("alpha", set()))
        self.assertIn("alpha", inbound.get("beta", set()))


if __name__ == "__main__":
    unittest.main()

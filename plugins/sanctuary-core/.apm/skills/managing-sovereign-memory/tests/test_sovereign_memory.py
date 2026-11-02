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
import record_procedure_outcome


class TestSovereignMemory(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.test_dir.name)
        (self.root / "LEARNING" / "concepts").mkdir(parents=True)
        (self.root / "LEARNING" / "procedures").mkdir(parents=True)
        (self.root / "context").mkdir(parents=True)
        (self.root / "protocols").mkdir(parents=True)
        (self.root / ".agent" / "learning").mkdir(parents=True)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_progressive_disclosure_summary(self):
        sample_file = self.root / "LEARNING" / "concepts" / "sample.md"
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

        result = disclose.extract_summary(sample_file, max_lines=20)
        self.assertEqual(result["status"], "ok")
        self.assertIn("This is the executive summary text", result["summary"])
        self.assertNotIn("Deep technical content", result["summary"])
        self.assertEqual(result["metadata"].get("title"), "Sample Note")

    def test_tier_normalization(self):
        self.assertEqual(disclose.normalize_tier("1", None), "index")
        self.assertEqual(disclose.normalize_tier("2", None), "summary")
        self.assertEqual(disclose.normalize_tier("3", None), "deep")
        self.assertEqual(disclose.normalize_tier(None, "index"), "index")

    def test_asymmetric_recording_and_event_journal(self):
        event = {
            "event_id": "test-uuid-1234",
            "timestamp": "2026-11-02T00:00:00Z",
            "session_id": "test-session",
            "topic": "test_topic",
            "type": "FRICTION",
            "insight": "Failed on edge case X",
            "evidence": "tests/test.py"
        }
        events_path = record_insight.append_event_log(self.root, event)
        self.assertTrue(events_path.exists())
        self.assertIn("test-uuid-1234", events_path.read_text(encoding="utf-8"))

        record_insight.record_to_context_memory(self.root, "test_topic", "friction", "Failed on edge case X")
        mem_file = self.root / "context" / "memory.md"
        self.assertTrue(mem_file.exists())
        self.assertIn("Failed on edge case X", mem_file.read_text(encoding="utf-8"))

        concept_file = record_insight.record_to_learning_concepts(self.root, "test_topic", "observed", "Discovered behavior Y")
        self.assertTrue(concept_file.exists())
        self.assertIn("Discovered behavior Y", concept_file.read_text(encoding="utf-8"))

    def test_confidence_decay(self):
        decayed_file = self.root / "LEARNING" / "concepts" / "old_concept.md"
        decayed_file.write_text("""# Old Concept
**Status:** CONFIRMED
**Last Verified:** 2025-01-01
""", encoding="utf-8")

        decay_count = audit_decay.audit_concepts(self.root, apply_fixes=True)
        self.assertEqual(decay_count, 1)
        self.assertIn("**Status:** OBSERVED", decayed_file.read_text(encoding="utf-8"))

    def test_graph_query_with_pipe_and_heading_normalization(self):
        note_a = self.root / "LEARNING" / "concepts" / "alpha.md"
        note_b = self.root / "LEARNING" / "concepts" / "beta.md"
        note_a.write_text("# Alpha\nRefers to [[beta#section|Beta Label]] and [link](beta.md#deep).", encoding="utf-8")
        note_b.write_text("# Beta\nTarget note.", encoding="utf-8")

        outbound, inbound, path_map = query_graph.build_graph(self.root)
        self.assertIn("beta", outbound.get("alpha", set()))
        self.assertIn("alpha", inbound.get("beta", set()))

    def test_procedure_outcome_recording(self):
        proc_file = self.root / "LEARNING" / "procedures" / "test_proc.md"
        proc_file.write_text("""---
id: proc-test-sample
name: Test Sample Procedure
version: 1.0.0
outcomes:
  environment: test-env
  success_count: 2
  failure_count: 1
  blocked_count: 0
---
# Test Procedure Body
""", encoding="utf-8")

        found = record_procedure_outcome.find_procedure_file(self.root, "proc-test-sample", None)
        self.assertIsNotNone(found)
        self.assertEqual(found, proc_file)

        fm_data, _, body = record_procedure_outcome.parse_frontmatter(proc_file.read_text(encoding="utf-8"))
        self.assertEqual(fm_data["id"], "proc-test-sample")
        self.assertEqual(fm_data["outcomes"]["success_count"], 2)


if __name__ == "__main__":
    unittest.main()

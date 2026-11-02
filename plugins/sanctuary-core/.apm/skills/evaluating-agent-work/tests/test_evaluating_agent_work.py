#!/usr/bin/env python3
"""
Unit tests for evaluating-agent-work skill scripts.
"""

import sys
import unittest
import tempfile
import json
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import score_contribution
import track_budget
import visualize_economy


class TestEvaluatingAgentWork(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.test_dir.name)
        (self.root / ".agent" / "learning").mkdir(parents=True)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_score_contribution_and_ledger_projection(self):
        assessment = {
            "assessment_id": "test-assess-1",
            "timestamp": "2026-11-02T00:00:00Z",
            "task_id": "task-test-01",
            "agent_id": "astra",
            "dimensions": {
                "useful_outcome": 3,
                "learning_depth": 3,
                "verification_rigor": 3,
                "token_economy": 2,
                "anti_asch_integrity": 3
            },
            "raw_score": 14,
            "max_score": 15,
            "cognitive_fitness": 0.933,
            "notes": "Excellent performance"
        }

        ledger_file = score_contribution.update_ledger_projection(self.root, assessment)
        self.assertTrue(ledger_file.exists())

        data = json.loads(ledger_file.read_text(encoding="utf-8"))
        self.assertEqual(data["total_assessments"], 1)
        self.assertEqual(data["cumulative_score"], 14)
        self.assertAlmostEqual(data["average_fitness"], 0.933, places=2)
        self.assertIn("astra", data["agents"])
        self.assertEqual(data["agents"]["astra"]["assessments"], 1)

    def test_track_budget_profiles(self):
        claude_prof = track_budget.BUDGET_PROFILES["claude-pro"]
        self.assertEqual(claude_prof["soft_turn_limit_per_task"], 12)

        data, ledger_path = track_budget.load_budget_ledger(self.root)
        self.assertEqual(data["total_tokens_consumed"], 0)

    def test_render_bar_visualizer(self):
        bar_full = visualize_economy.render_bar(1.0, 1.0, 10)
        self.assertEqual(bar_full, "██████████")

        bar_half = visualize_economy.render_bar(0.5, 1.0, 10)
        self.assertEqual(bar_half, "█████░░░░░")

        bar_zero = visualize_economy.render_bar(0.0, 1.0, 10)
        self.assertEqual(bar_zero, "░░░░░░░░░░")


if __name__ == "__main__":
    unittest.main()

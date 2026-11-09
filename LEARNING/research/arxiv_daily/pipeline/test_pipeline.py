"""Checks the arXiv daily outputs. Run: python3 LEARNING/research/arxiv_daily/pipeline/test_pipeline.py"""
import json
import sys
import time
import unittest
from pathlib import Path

DOSSIER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_digest  # noqa: E402


class TestArxivDaily(unittest.TestCase):
    def test_incoming(self):
        data = json.loads((DOSSIER / "incoming.json").read_text())
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        self.assertTrue(all({"id", "title", "summary"} <= set(p) for p in data))

    def test_latest_digest(self):
        text = (DOSSIER / "latest.md").read_text()
        self.assertIn("# arXiv Daily Digest", text)
        self.assertGreater(len(text), 200)

    def test_archive_has_todays_digest(self):
        self.assertTrue((DOSSIER / "digests" / f"{time.strftime('%Y-%m-%d')}.md").exists())

    def test_two_sentence_summary(self):
        self.assertEqual(make_digest.two_sentences("One. Two! Three?"), "One. Two!")

    def test_save_analysis_parsing(self):
        import save_analysis
        sample = ("Warning: preamble\n# Analysis\n## Relevance to existing missions\n- none\n## Impact on the project\n- x\n"
                  "## Improvement ideas\n- y\n## New mission seeds\n```jsonl\n"
                  '{"paper": "http://arxiv.org/abs/1", "question": "Does X hold?", "first_step": "try", "related_missions": ["MISSION-001"]}\n'
                  "not json\n```\n")
        self.assertTrue(save_analysis.clean(sample).startswith("# Analysis"))
        seeds = save_analysis.seeds_from(sample)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["related_missions"], ["MISSION-001"])


if __name__ == "__main__":
    unittest.main()

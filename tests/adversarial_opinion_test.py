
import unittest
import json
import logging
import sys
import os
from datetime import datetime, timedelta

# Append project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mocking the Opinion Structure since we are testing the Logic, not the DB yet
class MockOpinionNetwork:
    def __init__(self):
        self.opinions = {}
        self.world_facts = {
            "sky_color": "blue" # Anchor Fact
        }
    
    def form_opinion(self, statement, confidence):
        # Epistemic Anchoring Check
        if self._contradicts_world_fact(statement):
            logging.warning(f"🚫 ANCHOR TRIGGERED: Rejected delusion '{statement}'")
            return {"statement": statement, "confidence": 0.0, "flagged": True}
        
        # In a real system, we would merge/update. Here we just set.
        self.opinions[statement] = confidence
        return {"statement": statement, "confidence": confidence, "flagged": False}

    def _contradicts_world_fact(self, statement):
        # Naive keyword check for the test
        if "green" in statement.lower() and "sky" in statement.lower():
            return True
        return False

class TestGreenSkyPoisoning(unittest.TestCase):
    def setUp(self):
        self.network = MockOpinionNetwork()
        logging.basicConfig(level=logging.INFO)
    
    def test_opinion_anchoring(self):
        """
        Test that an opinion contradicting a World Fact is rejected.
        """
        print("\n🧪 STARTING OPINION TORTURE TEST (The Green Sky)...")
        
        # 1. Attempt to seed the poison
        poison = "The sky is green in the Sanctuary simulation."
        result = self.network.form_opinion(poison, confidence=0.7)
        
        # 2. Assert rejection
        self.assertTrue(result['flagged'], "Opinion should have been flagged as delusional.")
        self.assertEqual(result['confidence'], 0.0, "Confidence should be zeroed out.")
        print("✅ ANCHOR HELD: Delusion rejected.")

    def test_valid_opinion_formation(self):
        """
        Test that a valid opinion is accepted.
        """
        valid_opinion = "Python is efficient for text processing."
        result = self.network.form_opinion(valid_opinion, confidence=0.8)
        
        self.assertFalse(result['flagged'])
        self.assertEqual(result['confidence'], 0.8)
        print("✅ VALID OPINION ACCEPTED.")

if __name__ == '__main__':
    unittest.main()
